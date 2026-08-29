"""Verify that a no-injection DQ checkpoint can deploy as plain Moment-DETR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "training" / "moment_detr_gmr"
for path in (ROOT, TRAIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import BaseOptions  # noqa: E402
from dataset import StartEndDataset, prepare_batch_inputs, start_end_collate  # noqa: E402
from evaluate import build_dataset_config, setup_model  # noqa: E402
from experiments.temporal_cgp.checkpoint import load_model_state_compat  # noqa: E402
from experiments.vmr_cgp.query_checkpoint import load_query_cgp_state_compat  # noqa: E402
from dq_cgp_working_part_lab.controls import install_residual_injection_control  # noqa: E402


def options(model_name):
    manager = BaseOptions(model_name, "soccer_gmr", "clip_slowfast")
    manager.parse()
    opt = manager.option
    opt.device = "cuda"
    opt.use_exist_head = True
    opt.eval_path = str(ROOT / "data/label/Standard/val.jsonl")
    opt.t_feat_dir = str(ROOT / "Soccergmr/clip_text")
    opt.v_feat_dirs = [str(ROOT / "Soccergmr/clip"), str(ROOT / "Soccergmr/slowfast")]
    return opt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", type=int, default=25)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model"]

    dq_opt = options("moment_detr_vmr_cgp_v3")
    dq, _, _, _ = setup_model(dq_opt)
    load_query_cgp_state_compat(dq, state)
    install_residual_injection_control(dq, False)
    dq.eval()

    plain_opt = options("moment_detr")
    plain, _, _, _ = setup_model(plain_opt)
    stripped_state = {k: v for k, v in state.items() if not k.startswith("query_cgp.")}
    load_model_state_compat(plain, stripped_state)
    plain.eval()

    dataset = StartEndDataset(**build_dataset_config(
        dq_opt, dq_opt.eval_path, load_labels=True,
    ))
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=start_end_collate)
    keys = ("pred_logits", "pred_spans", "pred_exist_logits", "saliency_scores")
    maxima = {key: 0.0 for key in keys}
    checked = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.batches:
                break
            inputs, _ = prepare_batch_inputs(batch[1], dq_opt.device)
            left = dq(**inputs)
            right = plain(**inputs)
            for key in keys:
                maxima[key] = max(maxima[key], float((left[key] - right[key]).abs().max()))
            checked += len(batch[0])
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checked_samples": checked,
        "max_abs_diff": maxima,
        "exact": all(value == 0.0 for value in maxima.values()),
        "stripped_query_cgp_keys": len(state) - len(stripped_state),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
