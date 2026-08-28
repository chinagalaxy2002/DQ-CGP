"""Run fixed-K and decoder evidence-binding analysis for one checkpoint.

This entry point intentionally imports the repository implementation rather
than editing it.  Instrumentation and counterfactuals are installed only on
the in-memory model used by this process.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict
from torch.utils.data import DataLoader

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.temporal_cgp.checkpoint import load_model_state_compat
from experiments.vmr_cgp.query_ablation import apply_query_cgp_ablation
from experiments.vmr_cgp.query_checkpoint import (
    load_query_cgp_state_compat,
    restore_query_cgp_options,
)
from models.moment_detr_gmr.moment_detr import build_model
from models.moment_detr_gmr.utils.span_utils import span_cxw_to_xx
from occurrence_binding.bootstrap import paired_bootstrap
from occurrence_binding.capture_attention import (
    as_batch_head_query_key,
    get_decoder_attention,
    install_decoder_attention_capture,
)
from occurrence_binding.context_roll import apply_context_roll
from occurrence_binding.metrics import (
    attention_on_valid_video,
    binding_metrics,
    fixed_k_metrics,
    jsonable,
    route_metrics,
)
from training.moment_detr_gmr.config import BaseOptions
from training.moment_detr_gmr.dataset import (
    StartEndDataset,
    prepare_batch_inputs,
    start_end_collate,
)

LOGGER = logging.getLogger("occurrence_binding")
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def build_dataset_config(opt: Any, data_path: str) -> EasyDict:
    """Build the evaluation dataset arguments without importing evaluate.py."""

    return EasyDict(
        dset_name=opt.dset_name,
        domain=None,
        data_path=data_path,
        ctx_mode=opt.ctx_mode,
        v_feat_dirs=opt.v_feat_dirs,
        a_feat_dirs=None,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        v_feat_types=opt.v_feat_types,
        a_feat_types=None,
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        max_a_l=opt.max_a_l,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        load_labels=True,
        mr_only=True,
        keep_empty_gt=True,
        use_query_attention_mask=bool(getattr(opt, "use_query_attention_mask", False)),
        return_query_semantic_mask=bool(
            getattr(opt, "query_cgp_use_semantic_mask", False)
        ),
    )


def _value(opt: Any, key: str, default: Any = None) -> Any:
    if isinstance(opt, Mapping):
        return opt.get(key, default)
    return getattr(opt, key, default)


def _configure_options(
    model_name: str,
    split: str,
    eval_path: str,
    text_features: str,
    video_features: Sequence[str],
    device: str,
) -> Any:
    manager = BaseOptions(model_name, "soccer_gmr", "clip_slowfast")
    manager.parse()
    opt = manager.option
    opt.eval_path = eval_path
    opt.eval_split_name = split
    opt.t_feat_dir = text_features
    opt.v_feat_dirs = list(video_features)
    opt.device = device
    return opt


def _load_model(
    checkpoint_path: str,
    *,
    model_name: str,
    mode: str,
    split: str,
    eval_path: str,
    text_features: str,
    video_features: Sequence[str],
    device: str,
) -> tuple[torch.nn.Module, torch.nn.Module, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model")
    if state_dict is None:
        raise ValueError(f"checkpoint has no model state: {checkpoint_path}")

    opt = _configure_options(
        model_name,
        split,
        eval_path,
        text_features,
        video_features,
        device,
    )
    has_query_cgp = restore_query_cgp_options(opt, checkpoint)
    if has_query_cgp != model_name.endswith("v3"):
        LOGGER.info(
            "checkpoint architecture detected as %s (requested config=%s)",
            "DQ-CGP" if has_query_cgp else "baseline",
            model_name,
        )

    # The dataset config enables the existence head for Soccer-GMR, while the
    # released baseline checkpoint predates that head.  Follow the checkpoint
    # state as the source of truth before constructing the model.
    has_exist_head = any("exist_head" in key for key in state_dict)
    opt.use_exist_head = has_exist_head

    model, criterion = build_model(opt)
    if has_query_cgp:
        load_query_cgp_state_compat(model, state_dict)
    else:
        load_model_state_compat(model, state_dict)
    model.to(device)
    criterion.to(device)
    model.eval()
    criterion.eval()

    if mode == "dq_beta_zero":
        apply_query_cgp_ablation(model, "beta_zero")
    elif mode == "dq_context_roll":
        apply_context_roll(model)
    elif mode not in {"baseline", "dq_active"}:
        raise ValueError(f"unsupported analysis mode: {mode}")
    if mode.startswith("dq_") and getattr(model, "query_cgp", None) is None:
        raise ValueError(f"mode {mode} requires a DQ-CGP checkpoint")
    return model, criterion, opt


def _as_float_windows(windows: Any) -> list[list[float]]:
    if not isinstance(windows, list):
        return []
    return [[float(window[0]), float(window[1])] for window in windows]


def _prediction_data(outputs: Mapping[str, torch.Tensor], metas: Sequence[Mapping[str, Any]]) -> tuple[list[list[list[float]]], list[list[float]], list[list[int]]]:
    probabilities = F.softmax(outputs["pred_logits"], dim=-1)[..., 0].detach().cpu().numpy()
    spans = span_cxw_to_xx(outputs["pred_spans"]).detach().cpu().numpy()
    predictions: list[list[list[float]]] = []
    ranked_scores: list[list[float]] = []
    ranked_indices: list[list[int]] = []
    for batch_index, meta in enumerate(metas):
        duration = float(meta["duration"])
        seconds = np.clip(spans[batch_index] * duration, 0.0, duration)
        order = np.argsort(-probabilities[batch_index], kind="stable")
        predictions.append(
            [[float(seconds[index, 0]), float(seconds[index, 1])] for index in order]
        )
        ranked_scores.append([float(probabilities[batch_index, index]) for index in order])
        ranked_indices.append([int(index) for index in order])
    return predictions, ranked_scores, ranked_indices


def _target_indices(criterion: Any, outputs: Mapping[str, Any], targets: Mapping[str, Any], batch_index: int) -> tuple[list[int], list[int]]:
    assignments = criterion.matcher(outputs, targets)[batch_index]
    query_indices, gt_indices = assignments
    return (
        [int(value) for value in query_indices.detach().cpu().tolist()],
        [int(value) for value in gt_indices.detach().cpu().tolist()],
    )


def _binding_or_none(
    attention: np.ndarray | None,
    gt_windows: Sequence[Sequence[float]],
    query_indices: Sequence[int],
    gt_indices: Sequence[int],
    *,
    clip_length: float,
    duration: float,
) -> dict[str, Any] | None:
    return binding_metrics(
        attention,
        gt_windows,
        query_indices,
        gt_indices,
        clip_length=clip_length,
        duration=duration,
    )


def _summarize(records: Sequence[Mapping[str, Any]], mode: str, checkpoint: str, split: str) -> dict[str, Any]:
    def subset(predicate):
        return [record for record in records if predicate(record)]

    def mean(values):
        valid = [float(value) for value in values if value is not None]
        return float(np.mean(valid)) if valid else None

    def metric_summary(items):
        result: dict[str, Any] = {}
        for key in ("coverage_k", "duplicate_rate_k", "pairwise_iou_k"):
            all_keys = sorted({sub_key for item in items for sub_key in item[key]})
            result[key] = {
                sub_key: mean(item[key].get(sub_key) for item in items)
                for sub_key in all_keys
            }
        for layer in ("d1", "d2", "dq_private"):
            result[layer] = {
                metric: mean(
                    item[layer].get(metric) if item[layer] is not None else None
                    for item in items
                )
                for metric in ("aec", "binding_margin", "ecr", "own_mass")
            }
        result["route"] = {
            metric: mean(
                item["route"].get(metric) if item["route"] is not None else None
                for item in items
            )
            for metric in ("route_entropy", "route_entropy_std")
        }
        return result

    positive = subset(lambda record: record["num_gt"] > 0)
    multi = subset(lambda record: record["num_gt"] >= 2)
    buckets: dict[str, list[Mapping[str, Any]]] = {"all": records, "positive": positive, "multi_occurrence": multi}
    for upper in (1, 2, 3, 4):
        if upper == 1:
            buckets["gt_1"] = subset(lambda record: record["num_gt"] == 1)
        elif upper == 2:
            buckets["gt_2"] = subset(lambda record: record["num_gt"] == 2)
        elif upper == 3:
            buckets["gt_3_4"] = subset(lambda record: 3 <= record["num_gt"] <= 4)
        else:
            buckets["gt_5_plus"] = subset(lambda record: record["num_gt"] >= 5)

    return {
        "mode": mode,
        "checkpoint": str(checkpoint),
        "split": split,
        "num_records": len(records),
        "num_positive_records": len(positive),
        "num_multi_occurrence_records": len(multi),
        "num_empty_records": sum(record["num_gt"] == 0 for record in records),
        "buckets": {
            name: {"num_records": len(items), "metrics": metric_summary(items)}
            for name, items in buckets.items()
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode.startswith("dq_") and args.model_name == "moment_detr":
        args.model_name = "moment_detr_vmr_cgp_v3"
    model, criterion, opt = _load_model(
        args.checkpoint,
        model_name=args.model_name,
        mode=args.mode,
        split=args.split,
        eval_path=args.eval_path,
        text_features=args.text_features,
        video_features=args.video_features,
        device=args.device,
    )

    dataset_config = build_dataset_config(opt, args.eval_path)
    # Keep null-set queries in records; they are excluded automatically from
    # positive/multi-occurrence aggregate metrics.
    dataset_config.keep_empty_gt = True
    dataset = StartEndDataset(**dataset_config)
    loader = DataLoader(
        dataset,
        collate_fn=start_end_collate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    wrappers = install_decoder_attention_capture(model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    attention_dir = output_dir / "fig_data"
    attention_qids = {
        item.strip() for item in args.save_attention_qids.split(",") if item.strip()
    }
    records: list[dict[str, Any]] = []
    clip_length = float(getattr(opt, "clip_length", 2.0))

    with torch.no_grad():
        for batch_index, (metas, batch_inputs) in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            model_inputs, targets = prepare_batch_inputs(batch_inputs, args.device)
            outputs = model(**model_inputs)
            captured = [
                as_batch_head_query_key(attention, len(metas)).cpu().numpy()
                for attention in get_decoder_attention(wrappers)
            ]
            predictions, ranked_scores, ranked_indices = _prediction_data(outputs, metas)
            assignments = criterion.matcher(outputs, targets)
            private_attention = outputs.get("query_cgp_temporal_attention")
            private_weights = outputs.get("query_cgp_basis_weights")
            private_attention_np = (
                private_attention.detach().cpu().numpy()
                if private_attention is not None
                else None
            )
            private_weights_np = (
                private_weights.detach().cpu().numpy()
                if private_weights is not None
                else None
            )
            video_valid_batch = model_inputs["src_vid_mask"].detach().cpu().bool().numpy()

            for local_index, meta in enumerate(metas):
                gt_windows = _as_float_windows(meta.get("relevant_windows", []))
                duration = float(meta["duration"])
                query_indices, gt_indices = (
                    [int(value) for value in assignments[local_index][0].detach().cpu().tolist()],
                    [int(value) for value in assignments[local_index][1].detach().cpu().tolist()],
                )
                native_video = [
                    attention_on_valid_video(
                        captured[layer_index][local_index],
                        video_valid_batch[local_index],
                    )
                    for layer_index in range(len(captured))
                ]
                d1 = _binding_or_none(
                    native_video[0] if native_video else None,
                    gt_windows,
                    query_indices,
                    gt_indices,
                    clip_length=clip_length,
                    duration=duration,
                )
                d2 = _binding_or_none(
                    native_video[-1] if native_video else None,
                    gt_windows,
                    query_indices,
                    gt_indices,
                    clip_length=clip_length,
                    duration=duration,
                )

                private_video = None
                if private_attention_np is not None:
                    private_video = attention_on_valid_video(
                        private_attention_np[local_index],
                        video_valid_batch[local_index],
                    )
                dq_private = _binding_or_none(
                    private_video,
                    gt_windows,
                    query_indices,
                    gt_indices,
                    clip_length=clip_length,
                    duration=duration,
                )
                route = route_metrics(
                    private_weights_np[local_index] if private_weights_np is not None else None,
                    query_indices,
                )
                fixed = fixed_k_metrics(predictions[local_index], gt_windows)
                coverage_keys = [key for key in fixed if key.startswith("coverage@")]
                duplicate_keys = [key for key in fixed if key.startswith("duplicate_rate@")]
                pairwise_keys = [key for key in fixed if key.startswith("pairwise_iou@")]
                record: dict[str, Any] = {
                    "mode": args.mode,
                    "batch_index": batch_index,
                    "qid": int(meta["qid"]),
                    "vid": meta["vid"],
                    "query": meta.get("query"),
                    "duration": duration,
                    "num_gt": len(gt_windows),
                    "gt_windows": gt_windows,
                    "query_indices_ranked": ranked_indices[local_index],
                    "scores": ranked_scores[local_index],
                    "pred_windows": predictions[local_index],
                    "hungarian_query_idx": query_indices,
                    "hungarian_gt_idx": gt_indices,
                    "coverage_k": {
                        key.removeprefix("coverage@"): fixed[key] for key in coverage_keys
                    },
                    "duplicate_rate_k": {
                        key.removeprefix("duplicate_rate@"): fixed[key] for key in duplicate_keys
                    },
                    "pairwise_iou_k": {
                        key.removeprefix("pairwise_iou@"): fixed[key] for key in pairwise_keys
                    },
                    "d1": d1,
                    "d2": d2,
                    "dq_private": dq_private,
                    "route": route,
                }
                if private_attention_np is not None:
                    diagnostics = getattr(getattr(model, "query_cgp", None), "last_output", None)
                    if diagnostics is not None:
                        record["residual_update_l2"] = [
                            float(value)
                            for value in diagnostics.residual_update[local_index]
                            .detach()
                            .norm(dim=-1)
                            .cpu()
                            .tolist()
                        ]
                        record["residual_update_l2_mean"] = float(
                            np.mean(record["residual_update_l2"])
                        )
                qid_key = str(meta["qid"])
                if qid_key in attention_qids:
                    attention_dir.mkdir(parents=True, exist_ok=True)
                    npz_path = attention_dir / f"{qid_key}.npz"
                    np.savez_compressed(
                        npz_path,
                        native_d1=native_video[0],
                        native_d2=native_video[-1],
                        dq_private=(private_video if private_video is not None else np.empty((0, 0))),
                        gt_windows=np.asarray(gt_windows, dtype=np.float64),
                        pred_windows=np.asarray(predictions[local_index], dtype=np.float64),
                    )
                    record["attention_npz"] = str(npz_path)
                records.append(jsonable(record))

            if (batch_index + 1) % args.log_every == 0:
                LOGGER.info("processed %d/%d batches (%d records)", batch_index + 1, len(loader), len(records))

    records_path = output_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = _summarize(records, args.mode, args.checkpoint, args.split)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("wrote %s and %s", records_path, summary_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "dq_active", "dq_beta_zero", "dq_context_roll"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-name", default="moment_detr")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--eval-path", default=None)
    parser.add_argument("--text-features", default=str(REPO_ROOT / "Soccergmr" / "clip_text"))
    parser.add_argument(
        "--video-features",
        nargs=2,
        default=[str(REPO_ROOT / "Soccergmr" / "clip"), str(REPO_ROOT / "Soccergmr" / "slowfast")],
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-attention-qids", default="")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Process at most this many batches; useful for an interface smoke test.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.eval_path is None:
        args.eval_path = str(REPO_ROOT / "data" / "label" / "Standard" / f"{args.split}.jsonl")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    run(args)


if __name__ == "__main__":
    main()
