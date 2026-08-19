"""Measure how strongly a trained VMR-CGP checkpoint changes text features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.vmr_cgp.checkpoint import (
    load_vmr_cgp_state_compat,
    restore_vmr_cgp_options,
)
from models.moment_detr_gmr.moment_detr import build_model
from training.moment_detr_gmr.config import BaseOptions
from training.moment_detr_gmr.dataset import (
    StartEndDataset,
    prepare_batch_inputs,
    start_end_collate,
)


def summarize(values: torch.Tensor) -> dict[str, float]:
    values = values.float().cpu()
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p95": float(torch.quantile(values, 0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


@torch.no_grad()
def diagnose(args: argparse.Namespace) -> dict:
    manager = BaseOptions(args.model, "soccer_gmr", "clip_slowfast")
    manager.parse()
    opt = manager.option
    opt.device = args.device
    opt.t_feat_dir = args.text_features
    opt.v_feat_dirs = args.video_features

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not restore_vmr_cgp_options(opt, checkpoint):
        raise ValueError("checkpoint does not contain vmr_cgp.* parameters")
    model, _ = build_model(opt)
    load_vmr_cgp_state_compat(model, checkpoint["model"])
    model.to(args.device).eval()

    dataset = StartEndDataset(
        dset_name=opt.dset_name,
        domain=None,
        data_path=args.data_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        v_feat_types=opt.v_feat_types,
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        max_a_l=opt.max_a_l,
        ctx_mode=opt.ctx_mode,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        load_labels=True,
        mr_only=True,
        keep_empty_gt=True,
        use_query_attention_mask=bool(opt.use_query_attention_mask),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=start_end_collate,
    )

    ratios = []
    cosines = []
    entropies = []
    is_positive = []
    for _, batched_inputs in loader:
        model_inputs, targets = prepare_batch_inputs(batched_inputs, args.device)
        video = model.input_vid_proj(model_inputs["src_vid"])
        text = model.input_txt_proj(model_inputs["src_txt"])
        text_mask = model_inputs["src_txt_mask"].bool()
        output = model.vmr_cgp(
            video,
            model_inputs["src_vid_mask"],
            text,
            model_inputs["src_txt_mask"],
        )

        delta_norm = (output.enhanced_text - text).norm(dim=-1)
        text_norm = text.norm(dim=-1)
        mask = text_mask.to(text.dtype)
        ratios.append(
            (delta_norm * mask).sum(dim=1)
            / (text_norm * mask).sum(dim=1).clamp_min(1e-8)
        )
        token_cosine = F.cosine_similarity(output.enhanced_text, text, dim=-1)
        cosines.append((token_cosine * mask).sum(dim=1) / mask.sum(dim=1))

        weights = output.basis_weights.clamp_min(torch.finfo(text.dtype).eps)
        entropy = -(weights * weights.log()).sum(dim=-1)
        entropy = entropy / torch.log(
            torch.tensor(weights.shape[-1], device=weights.device, dtype=text.dtype)
        )
        entropies.append((entropy * mask).sum(dim=1) / mask.sum(dim=1))
        is_positive.append(targets["exist_label"].bool())

    ratios = torch.cat(ratios)
    cosines = torch.cat(cosines)
    entropies = torch.cat(entropies)
    is_positive = torch.cat(is_positive)
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "num_examples": len(dataset),
        "alpha": float(model.vmr_cgp.alpha.detach().cpu()),
        "alpha_trainable": bool(model.vmr_cgp.alpha.requires_grad),
        "gate_floor": float(model.vmr_cgp.gate_floor),
        "relative_feature_delta": summarize(ratios),
        "relative_feature_delta_positive": summarize(ratios[is_positive]),
        "relative_feature_delta_null": summarize(ratios[~is_positive]),
        "text_cosine_after_enhancement": summarize(cosines),
        "normalized_basis_entropy": summarize(entropies),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="moment_detr_vmr_cgp_v2")
    parser.add_argument("--data_path", default="data/label/Standard/val.jsonl")
    parser.add_argument("--text_features", default="Soccergmr/clip_text")
    parser.add_argument(
        "--video_features", nargs=2, default=["Soccergmr/clip", "Soccergmr/slowfast"]
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    diagnostics = diagnose(parsed)
    output_path = Path(parsed.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)
        handle.write("\n")
    print(json.dumps(diagnostics, indent=2))
