"""Training script for LS-DQ-CGP (Late-Semantic DQ-CGP)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "training" / "moment_detr_gmr"
for path in (ROOT, TRAIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import BaseOptions
from dataset import StartEndDataset
from evaluate import setup_model
from training.moment_detr_gmr.train import build_dataset_config, train
from models.moment_detr_gmr.moment_detr import build_model as build_base_model

from ls_dq_cgp_lab.ls_dq_cgp_model import LSDQCGPModel, install_ls_dq_cgp_loss

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train LS-DQ-CGP on Soccer-GMR")
    parser.add_argument("--output", default="outputs/ls_dq_cgp_seed2023", help="Output directory")
    parser.add_argument("--seed", type=int, default=2023, help="Random seed")
    parser.add_argument("--device", default="cuda", help="Computation device")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    parser.add_argument("--epochs", type=int, default=400, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--native_bind_coef", type=float, default=0.2, help="Coefficient for D1 binding loss")
    parser.add_argument("--num_basis", type=int, default=16, help="Number of learnable basis prompts")
    parser.add_argument("--prompt_length", type=int, default=6, help="Prompt token length")
    existence = parser.add_mutually_exclusive_group()
    existence.add_argument("--use_exist_head", action="store_true", help="Enable GMR existence head")
    existence.add_argument("--no_exist_head", action="store_false", dest="use_exist_head", help="Disable GMR existence head")
    parser.set_defaults(use_exist_head=False)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists and is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    # Initialize BaseOptions
    manager = BaseOptions("moment_detr", "soccer_gmr", "clip_slowfast")
    manager.parse()
    opt = manager.option

    opt.seed = args.seed
    opt.device = args.device
    opt.n_epoch = args.epochs
    opt.lr = args.lr
    opt.results_dir = str(output)
    opt.ckpt_filepath = str(output / opt.ckpt_filename)
    opt.train_log_filepath = str(output / opt.train_log_filename)
    opt.eval_log_filepath = str(output / opt.eval_log_filename)
    opt.train_path = str(ROOT / "data/label/Standard/train.jsonl")
    opt.eval_path = str(ROOT / "data/label/Standard/val.jsonl")
    opt.t_feat_dir = str(ROOT / "Soccergmr/clip_text")
    opt.v_feat_dirs = [str(ROOT / "Soccergmr/clip"), str(ROOT / "Soccergmr/slowfast")]
    opt.mr_only = not args.use_exist_head
    opt.lw_saliency = 1
    opt.use_exist_head = bool(args.use_exist_head)
    opt.exist_loss_coef = 1.0
    opt.exist_gate_thd = 0.3
    opt.exist_pool = "max"
    opt.query_cgp_use_semantic_mask = True
    opt.component_variant = "ls_dq_cgp_exist" if args.use_exist_head else "ls_dq_cgp"

    experiment_meta = {
        "variant": opt.component_variant,
        "seed": args.seed,
        "model": "LateSemantic_MomentDETR",
        "use_exist_head": args.use_exist_head,
        "native_binding_coef": args.native_bind_coef,
        "num_basis": args.num_basis,
        "prompt_length": args.prompt_length,
        "query_cgp_use_semantic_mask": True,
        "epochs": args.epochs,
        "lr": args.lr,
    }
    (output / "experiment.json").write_text(json.dumps(experiment_meta, indent=2) + "\n")

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Prepare datasets (keep_empty_gt=True for training when exist head is active)
    train_dataset = StartEndDataset(
        **build_dataset_config(opt, opt.train_path, load_labels=True, keep_empty_gt=bool(args.use_exist_head))
    )
    val_dataset = StartEndDataset(
        **build_dataset_config(opt, opt.eval_path, load_labels=True, keep_empty_gt=False)
    )

    # Build base model and wrap with LS-DQ-CGP
    base_model, criterion = build_base_model(opt)
    model = LSDQCGPModel(
        base_model=base_model,
        num_basis=args.num_basis,
        prompt_length=args.prompt_length,
        router_hidden_dim=256,
        frf_hidden_dim=512,
        temperature=1.0,
    )
    install_ls_dq_cgp_loss(criterion, model, coefficient=args.native_bind_coef)

    model.to(opt.device)
    criterion.to(opt.device)

    # Setup optimizer and scheduler
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad]}
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=opt.lr, weight_decay=opt.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop)

    logger.info(f"Starting LS-DQ-CGP training with exist_head={args.use_exist_head}, semantic_mask={opt.query_cgp_use_semantic_mask}, seed={args.seed}, total epochs={args.epochs}...")
    train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt)


if __name__ == "__main__":
    main()
