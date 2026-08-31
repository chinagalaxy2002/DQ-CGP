"""Evaluate a semantic ablation with a Full or retrained checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "training" / "moment_detr_gmr"
for path in (ROOT, TRAIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import BaseOptions
from dataset import StartEndDataset
from eval.eval_main import evaluate_gmr
from evaluate import build_dataset_config, eval_epoch
from models.moment_detr_gmr.moment_detr import build_model as build_base_model
from ls_dq_cgp_ablation_lab.ablation_model import (
    ABLATION_VARIANTS,
    AblationLSDQCGPModel,
)
from ls_dq_cgp_ablation_lab.common import load_jsonl, portable_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=ABLATION_VARIANTS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", choices=["inference_only", "retrained"], required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval_bsz", type=int, default=4)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    has_exist_head = any("exist_head" in key for key in state_dict)

    manager = BaseOptions("moment_detr", "soccer_gmr", "clip_slowfast")
    manager.parse()
    opt = manager.option
    opt.device = args.device
    opt.eval_bsz = args.eval_bsz
    opt.results_dir = str(output)
    opt.eval_split_name = args.split
    label_path = ROOT / f"data/label/Standard/{args.split}.jsonl"
    opt.eval_path = str(label_path)
    opt.t_feat_dir = str(ROOT / "Soccergmr/clip_text")
    opt.v_feat_dirs = [str(ROOT / "Soccergmr/clip"), str(ROOT / "Soccergmr/slowfast")]
    opt.query_cgp_use_semantic_mask = True
    opt.use_exist_head = has_exist_head
    opt.mr_only = not has_exist_head
    opt.exist_loss_coef = 1.0
    opt.exist_gate_thd = 0.3
    opt.exist_pool = "max"
    opt.lw_saliency = 0

    dataset_config = build_dataset_config(opt, opt.eval_path, load_labels=False)
    if args.split == "test" or has_exist_head:
        dataset_config.keep_empty_gt = True
    dataset = StartEndDataset(**dataset_config)
    base_model, _ = build_base_model(opt)
    model = AblationLSDQCGPModel(
        base_model=base_model,
        num_basis=16,
        prompt_length=6,
        router_hidden_dim=256,
        frf_hidden_dim=512,
        temperature=1.0,
        variant=args.variant,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.to(opt.device).eval()

    save_filename = f"moment_detr_gmr_{args.split}_submission.jsonl"
    with torch.no_grad():
        eval_epoch(None, model, dataset, opt, save_filename, criterion=None)
    metrics = evaluate_gmr(
        load_jsonl(output / save_filename), load_jsonl(label_path),
        map_num_workers=1, verbose=False,
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result = {
        "variant": args.variant,
        "protocol": args.protocol,
        "has_exist_head": has_exist_head,
        "split": args.split,
        "checkpoint": portable_path(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)) + 1,
        "brief": dict(metrics["brief"]),
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

