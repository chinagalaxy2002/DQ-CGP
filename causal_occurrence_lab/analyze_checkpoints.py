"""Analyze existing checkpoints and causal-training checkpoints.

This script evaluates D1 auxiliary predictions separately from D2, evaluates
both D1 matching definitions, and emits raw/normalized occurrence evidence.
It is intentionally independent of the existing ``occurrence_binding_lab``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from causal_occurrence_lab.common import (
    REPO_ROOT,
    as_windows,
    build_dataset_config,
    load_model_for_analysis,
    mean_valid,
    prediction_data,
    submission_for_layer,
    write_json,
)
from causal_occurrence_lab.metrics import (
    attention_on_valid_video,
    binding_metrics,
    clean_multi_occurrence,
    fixed_k_metrics,
    jsonable,
    route_metrics,
)
from occurrence_binding_lab.occurrence_binding.capture_attention import (
    as_batch_head_query_key,
    get_decoder_attention,
    install_decoder_attention_capture,
)
from training.moment_detr_gmr.dataset import prepare_batch_inputs, start_end_collate

from eval.eval_main import evaluate_gmr

LOGGER = logging.getLogger("causal_occurrence_lab.analysis")


def _lists(assignments: tuple[torch.Tensor, torch.Tensor]) -> tuple[list[int], list[int]]:
    return (
        [int(item) for item in assignments[0].detach().cpu().tolist()],
        [int(item) for item in assignments[1].detach().cpu().tolist()],
    )


def _prediction_metrics(
    records: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    layer: str,
    map_workers: int,
) -> dict[str, Any]:
    submission = submission_for_layer(records, layer)
    result = evaluate_gmr(
        submission,
        list(ground_truth),
        k_list=(1, 3, 5),
        max_pred_windows=10,
        map_num_workers=map_workers,
        verbose=False,
    )
    return jsonable(dict(result))


def _aggregate_route(items: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any] | None:
    route_items = [item.get(key) for item in items if item.get(key) is not None]
    if not route_items:
        return None
    marginal = np.zeros(len(route_items[0]["marginal_usage"]), dtype=np.float64)
    argmax_usage = np.zeros(len(marginal), dtype=np.int64)
    total = 0
    conditional = []
    for item in route_items:
        count = int(item.get("num_queries", 0))
        marginal += np.asarray(item["marginal_usage"], dtype=np.float64) * count
        argmax_usage += np.asarray(item.get("argmax_usage", [0] * len(marginal)), dtype=np.int64)
        total += count
        conditional.extend([float(item["route_entropy"])] * max(count, 1))
    marginal /= max(total, 1)
    eps = np.finfo(np.float64).eps
    entropy = float(-(marginal * np.log(np.maximum(marginal, eps))).sum())
    denom = np.log(len(marginal)) if len(marginal) > 1 else 1.0
    return {
        "conditional_entropy_mean": float(np.mean(conditional)) if conditional else None,
        "conditional_entropy_std": float(np.std(conditional)) if conditional else None,
        "marginal_usage": marginal.tolist(),
        "marginal_entropy": entropy,
        "marginal_entropy_norm": entropy / denom,
        "effective_basis_count": float(np.exp(entropy)),
        "argmax_usage": argmax_usage.tolist(),
    }


def _mean_metric(items: Sequence[Mapping[str, Any]], path: Sequence[str]) -> float | None:
    values: list[Any] = []
    for item in items:
        current: Any = item
        for name in path:
            if not isinstance(current, Mapping):
                current = None
                break
            current = current.get(name)
        values.append(current)
    return mean_valid(values)


def _metric_summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "coverage_k", "d1_coverage_k", "duplicate_rate_k", "d1_duplicate_rate_k",
        "pairwise_iou_k", "d1_pairwise_iou_k",
    ):
        all_keys = sorted({sub_key for item in items for sub_key in item.get(key, {})})
        result[key] = {
            sub_key: mean_valid([item.get(key, {}).get(sub_key) for item in items])
            for sub_key in all_keys
        }
    for layer in ("d1_own", "d1_final", "d2", "dq_private"):
        result[layer] = {
            metric: _mean_metric(items, (layer, metric))
            for metric in (
                "aec", "aec_norm", "binding_margin", "binding_margin_norm",
                "ecr", "ecr_norm", "own_mass", "own_mass_norm",
            )
        }
    result["route_matched"] = _aggregate_route(items, "route_matched")
    result["route_all"] = _aggregate_route(items, "route_all")
    return result


def _make_buckets(records: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    return {
        "all": list(records),
        "positive": [r for r in records if r["num_gt"] > 0],
        "single": [r for r in records if r["num_gt"] == 1],
        "multi_occurrence": [r for r in records if r["num_gt"] >= 2],
        "two_occurrences": [r for r in records if r["num_gt"] == 2],
        "three_or_more_occurrences": [r for r in records if r["num_gt"] >= 3],
        "clean_multi_occurrence": [r for r in records if r["clean_multi_occurrence"]],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    eval_path = args.eval_path or str(REPO_ROOT / "data" / "label" / "Standard" / f"{args.split}.jsonl")
    model, criterion, opt, checkpoint = load_model_for_analysis(
        args.checkpoint,
        mode=args.mode,
        model_name=args.model_name,
        split=args.split,
        eval_path=eval_path,
        text_features=args.text_features,
        video_features=args.video_features,
        device=args.device,
    )
    dataset = __import__("training.moment_detr_gmr.dataset", fromlist=["StartEndDataset"]).StartEndDataset(
        **build_dataset_config(opt, eval_path)
    )
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
    records: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch_index, (metas, batch_inputs) in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            model_inputs, targets = prepare_batch_inputs(batch_inputs, args.device)
            outputs = model(**model_inputs)
            if "aux_outputs" not in outputs or not outputs["aux_outputs"]:
                raise RuntimeError("D1 evaluation requires aux_outputs[0]")
            d1_outputs = outputs["aux_outputs"][0]
            captured = [
                as_batch_head_query_key(attn, len(metas)).cpu().numpy()
                for attn in get_decoder_attention(wrappers)
            ]
            d1_predictions, d1_scores, d1_ranked = prediction_data(d1_outputs, metas)
            d2_predictions, d2_scores, d2_ranked = prediction_data(outputs, metas)
            d1_assignments = criterion.matcher(d1_outputs, targets)
            d2_assignments = criterion.matcher(outputs, targets)
            private_attention = outputs.get("query_cgp_temporal_attention")
            private_weights = outputs.get("query_cgp_basis_weights")
            private_attention_np = private_attention.detach().cpu().numpy() if private_attention is not None else None
            private_weights_np = private_weights.detach().cpu().numpy() if private_weights is not None else None
            valid_video = model_inputs["src_vid_mask"].detach().cpu().bool().numpy()
            diagnostics = getattr(getattr(model, "query_cgp", None), "last_output", None)

            for local_index, meta in enumerate(metas):
                gt_windows = as_windows(meta.get("relevant_windows", []))
                duration = float(meta["duration"])
                d1_own_q, d1_own_g = _lists(d1_assignments[local_index])
                d2_q, d2_g = _lists(d2_assignments[local_index])
                native_video = [
                    attention_on_valid_video(captured[layer][local_index], valid_video[local_index])
                    for layer in range(len(captured))
                ]
                d1_own = binding_metrics(
                    native_video[0], gt_windows, d1_own_q, d1_own_g,
                    clip_length=float(opt.clip_length), duration=duration,
                ) if native_video else None
                d1_final = binding_metrics(
                    native_video[0], gt_windows, d2_q, d2_g,
                    clip_length=float(opt.clip_length), duration=duration,
                ) if native_video else None
                d2 = binding_metrics(
                    native_video[-1], gt_windows, d2_q, d2_g,
                    clip_length=float(opt.clip_length), duration=duration,
                ) if native_video else None
                private_video = None
                if private_attention_np is not None:
                    private_video = attention_on_valid_video(
                        private_attention_np[local_index], valid_video[local_index]
                    )
                dq_private = binding_metrics(
                    private_video, gt_windows, d2_q, d2_g,
                    clip_length=float(opt.clip_length), duration=duration,
                )
                route_all = route_metrics(
                    private_weights_np[local_index] if private_weights_np is not None else None
                )
                route_matched = route_metrics(
                    private_weights_np[local_index] if private_weights_np is not None else None,
                    d2_q,
                )
                d2_fixed = fixed_k_metrics(d2_predictions[local_index], gt_windows)
                d1_fixed = fixed_k_metrics(d1_predictions[local_index], gt_windows)
                pred_exist = None
                if "pred_exist_logits" in outputs:
                    pred_exist = float(torch.sigmoid(outputs["pred_exist_logits"][local_index]).detach().cpu())

                record: dict[str, Any] = {
                    "mode": args.mode,
                    "batch_index": batch_index,
                    "qid": int(meta["qid"]),
                    "vid": meta["vid"],
                    "query": meta.get("query"),
                    "duration": duration,
                    "num_gt": len(gt_windows),
                    "gt_windows": gt_windows,
                    "clean_multi_occurrence": clean_multi_occurrence(gt_windows, args.clean_iou),
                    "d1_pred_windows": d1_predictions[local_index],
                    "d1_scores": d1_scores[local_index],
                    "d1_query_indices_ranked": d1_ranked[local_index],
                    "d2_pred_windows": d2_predictions[local_index],
                    "d2_scores": d2_scores[local_index],
                    "d2_query_indices_ranked": d2_ranked[local_index],
                    "pred_exist_score": pred_exist,
                    "coverage_k": {
                        key.removeprefix("coverage@"): d2_fixed[key]
                        for key in d2_fixed if key.startswith("coverage@")
                    },
                    "d1_coverage_k": {
                        key.removeprefix("coverage@"): d1_fixed[key]
                        for key in d1_fixed if key.startswith("coverage@")
                    },
                    "duplicate_rate_k": {
                        key.removeprefix("duplicate_rate@"): d2_fixed[key]
                        for key in d2_fixed if key.startswith("duplicate_rate@")
                    },
                    "d1_duplicate_rate_k": {
                        key.removeprefix("duplicate_rate@"): d1_fixed[key]
                        for key in d1_fixed if key.startswith("duplicate_rate@")
                    },
                    "pairwise_iou_k": {
                        key.removeprefix("pairwise_iou@"): d2_fixed[key]
                        for key in d2_fixed if key.startswith("pairwise_iou@")
                    },
                    "d1_pairwise_iou_k": {
                        key.removeprefix("pairwise_iou@"): d1_fixed[key]
                        for key in d1_fixed if key.startswith("pairwise_iou@")
                    },
                    "d1_own": d1_own,
                    "d1_final": d1_final,
                    "d2": d2,
                    "dq_private": dq_private,
                    "route_all": route_all,
                    "route_matched": route_matched,
                    "hungarian_d1_query_idx": d1_own_q,
                    "hungarian_d1_gt_idx": d1_own_g,
                    "hungarian_d2_query_idx": d2_q,
                    "hungarian_d2_gt_idx": d2_g,
                    "active_query_count": {
                        str(threshold): int(sum(score > threshold for score in d2_scores[local_index]))
                        for threshold in args.activity_thresholds
                    },
                }
                if diagnostics is not None:
                    residual = diagnostics.residual_update[local_index].detach()
                    adapted = diagnostics.adapted_state[:, local_index].detach()
                    beta = float(getattr(model.query_cgp, "beta", torch.tensor(0.0)).detach().cpu())
                    inject = bool(getattr(model.query_cgp, "_causal_inject_residual", True))
                    candidate = adapted - beta * residual if inject else adapted
                    update = beta * residual if inject else torch.zeros_like(residual)
                    record["residual_update_l2"] = residual.norm(dim=-1).cpu().tolist()
                    record["would_be_injected_update_l2"] = (beta * residual).norm(dim=-1).cpu().tolist()
                    record["injected_update_l2"] = update.norm(dim=-1).cpu().tolist()
                    record["relative_update"] = (
                        update.norm(dim=-1) / candidate.norm(dim=-1).clamp_min(1e-12)
                    ).cpu().tolist()
                    record["relative_update_mean"] = float(
                        np.mean(record["relative_update"])
                    )
                if args.save_attention_qids and str(meta["qid"]) in args.save_attention_qids:
                    fig_dir = output_dir / "fig_data"
                    fig_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        fig_dir / f"{meta['qid']}.npz",
                        native_d1=native_video[0] if native_video else np.empty((0, 0)),
                        native_d2=native_video[-1] if native_video else np.empty((0, 0)),
                        dq_private=private_video if private_video is not None else np.empty((0, 0)),
                        gt_windows=np.asarray(gt_windows, dtype=np.float64),
                        d1_pred_windows=np.asarray(d1_predictions[local_index], dtype=np.float64),
                        d2_pred_windows=np.asarray(d2_predictions[local_index], dtype=np.float64),
                    )
                records.append(jsonable(record))
            if (batch_index + 1) % args.log_every == 0:
                LOGGER.info("processed %d/%d batches (%d records)", batch_index + 1, len(loader), len(records))

    records_path = output_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    for layer in ("d1", "d2"):
        with (output_dir / f"submission_{layer}.jsonl").open("w", encoding="utf-8") as handle:
            for item in submission_for_layer(records, layer):
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    buckets = _make_buckets(records)
    summary = {
        "mode": args.mode,
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "num_records": len(records),
        "num_positive_records": sum(record["num_gt"] > 0 for record in records),
        "num_multi_occurrence_records": sum(record["num_gt"] >= 2 for record in records),
        "prediction_metrics": {
            "d1": _prediction_metrics(records, dataset.data, "d1", args.map_workers),
            "d2": _prediction_metrics(records, dataset.data, "d2", args.map_workers),
        },
        "buckets": {
            name: {"num_records": len(items), "metrics": _metric_summary(items)}
            for name, items in buckets.items()
        },
        "control": {
            "duplicate_attribution": "per-prediction argmax IoU with thresholded valid hits",
            "clean_iou_threshold": args.clean_iou,
            "length_normalization": "GT overlapping valid clip fraction",
        },
    }
    write_json(output_dir / "summary.json", summary)
    LOGGER.info("wrote %s and %s", records_path, output_dir / "summary.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "dq_active", "dq_beta_zero", "dq_stripped"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-name", default="auto")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--eval-path", default=None)
    parser.add_argument("--text-features", default=str(REPO_ROOT / "Soccergmr" / "clip_text"))
    parser.add_argument("--video-features", nargs=2, default=[str(REPO_ROOT / "Soccergmr" / "clip"), str(REPO_ROOT / "Soccergmr" / "slowfast")])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--map-workers", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--clean-iou", type=float, default=0.1)
    parser.add_argument("--activity-thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--save-attention-qids", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.save_attention_qids = {item.strip() for item in args.save_attention_qids.split(",") if item.strip()}
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    run(args)


if __name__ == "__main__":
    main()
