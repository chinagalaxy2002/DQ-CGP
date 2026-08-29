"""Evaluate a trained component variant with the official GMR metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "training" / "moment_detr_gmr"
for path in (ROOT, TRAIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import BaseOptions  # noqa: E402
from dataset import StartEndDataset  # noqa: E402
from eval.eval_main import evaluate_gmr  # noqa: E402
from evaluate import build_dataset_config, eval_epoch, setup_model  # noqa: E402
from experiments.temporal_cgp.checkpoint import load_model_state_compat  # noqa: E402
from experiments.vmr_cgp.query_checkpoint import (  # noqa: E402
    load_query_cgp_state_compat,
    restore_query_cgp_options,
)

from dq_cgp_working_part_lab.controls import install_residual_injection_control  # noqa: E402


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    saved_opt = checkpoint.get("opt")
    variant = getattr(saved_opt, "component_variant", "unknown")
    inject = bool(getattr(saved_opt, "component_inject_residual", True))
    has_dq = any(key.startswith("query_cgp.") for key in state)

    manager = BaseOptions(
        "moment_detr_vmr_cgp_v3" if has_dq else "moment_detr",
        "soccer_gmr", "clip_slowfast",
    )
    manager.parse()
    opt = manager.option
    if has_dq:
        restore_query_cgp_options(opt, checkpoint)
    opt.use_exist_head = any("exist_head" in key for key in state)
    opt.device = args.device
    opt.eval_split_name = args.split
    label_path = ROOT / "data" / "label" / "Standard" / f"{args.split}.jsonl"
    opt.eval_path = str(label_path)
    opt.t_feat_dir = str(ROOT / "Soccergmr" / "clip_text")
    opt.v_feat_dirs = [str(ROOT / "Soccergmr" / "clip"), str(ROOT / "Soccergmr" / "slowfast")]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    opt.results_dir = str(output)

    dataset = StartEndDataset(**build_dataset_config(opt, str(label_path), load_labels=False))
    model, _, _, _ = setup_model(opt)
    if has_dq:
        load_query_cgp_state_compat(model, state)
        install_residual_injection_control(model, inject)
    else:
        load_model_state_compat(model, state)
    model.eval()

    submission_name = f"moment_detr_gmr_{args.split}_submission.jsonl"
    with torch.no_grad():
        eval_epoch(None, model, dataset, opt, submission_name, criterion=None)
    submission_path = output / submission_name
    metrics = evaluate_gmr(
        load_jsonl(submission_path), load_jsonl(label_path),
        map_num_workers=1, verbose=False,
    )
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    record = {
        "variant": variant,
        "inject_residual": inject,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)) + 1,
        "brief": dict(metrics["brief"]),
    }
    (output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
