"""Evaluation script for LS-DQ-CGP (supports active, static_bypass, and context_roll modes with official GMR metrics)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
import pprint

import torch

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "training" / "moment_detr_gmr"
for path in (ROOT, TRAIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import BaseOptions
from dataset import StartEndDataset
from eval.eval_main import evaluate_gmr
from evaluate import eval_epoch, build_dataset_config
from models.moment_detr_gmr.moment_detr import build_model as build_base_model
from ls_dq_cgp_lab.ls_dq_cgp_model import LSDQCGPModel

logger = logging.getLogger(__name__)


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Evaluate LS-DQ-CGP")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .ckpt")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Evaluation split")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    parser.add_argument("--device", default="cuda", help="Computation device")
    counterfactual = parser.add_mutually_exclusive_group()
    counterfactual.add_argument("--static_bypass", action="store_true", help="Enable static text bypass counterfactual")
    counterfactual.add_argument("--context_roll", action="store_true", help="Enable context roll permutation counterfactual")
    parser.add_argument("--eval_bsz", type=int, default=4, help="Evaluation batch size")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    # Load checkpoint state first to detect exist_head
    ckpt_path = Path(args.checkpoint).resolve()
    logger.info(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt

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
    if has_exist_head:
        opt.exist_loss_coef = 1.0
        opt.exist_gate_thd = 0.3
        opt.exist_pool = "max"
        opt.mr_only = False
    else:
        opt.mr_only = True
    opt.lw_saliency = 0

    if args.static_bypass:
        mode_name = "static_bypass"
    elif args.context_roll:
        mode_name = "context_roll"
    else:
        mode_name = "active"

    save_filename = f"moment_detr_gmr_{args.split}_submission.jsonl"

    # Dataset configuration
    dset_config = build_dataset_config(opt, opt.eval_path, load_labels=False)
    if args.split == "test" or has_exist_head:
        dset_config.keep_empty_gt = True
    eval_dataset = StartEndDataset(**dset_config)

    # Build model with exist_head if checkpoint had it
    base_model, _ = build_base_model(opt)
    model = LSDQCGPModel(
        base_model=base_model,
        num_basis=16,
        prompt_length=6,
        router_hidden_dim=256,
        frf_hidden_dim=512,
        temperature=1.0,
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Loaded checkpoint (epoch {ckpt.get('epoch', -1)}, exist_head={has_exist_head}). Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    model.to(opt.device)
    model.eval()
    model.static_bypass = args.static_bypass
    model.context_roll = args.context_roll

    logger.info(f"Running inference on {args.split} (Mode: {mode_name})...")
    with torch.no_grad():
        eval_epoch(None, model, eval_dataset, opt, save_filename, criterion=None)

    submission_path = output / save_filename
    logger.info(f"Evaluating submission with official GMR benchmark metrics against {label_path}...")
    metrics = evaluate_gmr(
        load_jsonl(submission_path),
        load_jsonl(label_path),
        map_num_workers=1,
        verbose=False,
    )

    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    
    result_record = {
        "variant": "ls_dq_cgp_exist" if has_exist_head else "ls_dq_cgp",
        "has_exist_head": has_exist_head,
        "mode": mode_name,
        "split": args.split,
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)) + 1,
        "brief": dict(metrics["brief"]),
    }
    (output / "result.json").write_text(json.dumps(result_record, indent=2) + "\n")
    
    print("\n" + "=" * 60)
    print(f"LS-DQ-CGP Evaluation Summary [{args.split.upper()}] (Mode: {mode_name}, Exist Head: {has_exist_head})")
    print("=" * 60)
    for k, v in metrics["brief"].items():
        print(f"{k:15s}: {v}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
