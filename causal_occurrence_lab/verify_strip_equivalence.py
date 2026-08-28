"""Numerically verify DQ beta-zero and stripped-model equivalence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from causal_occurrence_lab.common import (
    REPO_ROOT,
    build_dataset_config,
    load_model_for_analysis,
)
from training.moment_detr_gmr.dataset import StartEndDataset, prepare_batch_inputs, start_end_collate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--eval-path", default=None)
    parser.add_argument("--text-features", default=str(REPO_ROOT / "Soccergmr" / "clip_text"))
    parser.add_argument("--video-features", nargs=2, default=[str(REPO_ROOT / "Soccergmr" / "clip"), str(REPO_ROOT / "Soccergmr" / "slowfast")])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=1, help="Use one CPU thread for reproducible bitwise comparison.")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    eval_path = args.eval_path or str(REPO_ROOT / "data" / "label" / "Standard" / f"{args.split}.jsonl")
    beta_zero, criterion, opt, _ = load_model_for_analysis(
        args.checkpoint, mode="dq_beta_zero", model_name="auto", split=args.split,
        eval_path=eval_path, text_features=args.text_features,
        video_features=args.video_features, device=args.device,
    )
    stripped, _, _, _ = load_model_for_analysis(
        args.checkpoint, mode="dq_stripped", model_name="auto", split=args.split,
        eval_path=eval_path, text_features=args.text_features,
        video_features=args.video_features, device=args.device,
    )
    dataset = StartEndDataset(**build_dataset_config(opt, eval_path))
    rng = np.random.default_rng(args.seed)
    selected = rng.choice(len(dataset), size=min(args.num_samples, len(dataset)), replace=False).tolist()
    loader = DataLoader(Subset(dataset, selected), collate_fn=start_end_collate, batch_size=args.batch_size, shuffle=False)
    maxima = {"pred_logits": 0.0, "pred_spans": 0.0, "d1_pred_logits": 0.0, "d1_pred_spans": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            inputs, _ = prepare_batch_inputs(batch[1], args.device)
            left, right = beta_zero(**inputs), stripped(**inputs)
            maxima["pred_logits"] = max(maxima["pred_logits"], float((left["pred_logits"] - right["pred_logits"]).abs().max()))
            maxima["pred_spans"] = max(maxima["pred_spans"], float((left["pred_spans"] - right["pred_spans"]).abs().max()))
            maxima["d1_pred_logits"] = max(maxima["d1_pred_logits"], float((left["aux_outputs"][0]["pred_logits"] - right["aux_outputs"][0]["pred_logits"]).abs().max()))
            maxima["d1_pred_spans"] = max(maxima["d1_pred_spans"], float((left["aux_outputs"][0]["pred_spans"] - right["aux_outputs"][0]["pred_spans"]).abs().max()))
            count += len(batch[0])
    result = {"checkpoint": args.checkpoint, "num_samples": count, "max_abs_difference": maxima, "pass_at_1e-6": all(value < 1e-6 for value in maxima.values())}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
