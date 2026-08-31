"""Train the strict ``Delta E_q = 0`` LS-DQ-CGP control from scratch.

This is a separate experimental entry point.  It does not modify the original
LS-DQ-CGP implementation or training script.
"""

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

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "training" / "moment_detr_gmr"
for path in (ROOT, TRAIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import BaseOptions
from dataset import StartEndDataset
from models.moment_detr_gmr.moment_detr import build_model as build_base_model
from training.moment_detr_gmr.train import build_dataset_config, train
from ls_dq_cgp_lab.ls_dq_cgp_model import install_ls_dq_cgp_loss
from strict_delta_zero_lab.strict_delta_zero_model import (
    StrictDeltaZeroLSDQCGPModel,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train strict Delta E_q = 0 LS-DQ-CGP from scratch"
    )
    parser.add_argument(
        "--output",
        default="outputs/ls_dq_cgp_delta_zero_train_seed2023",
    )
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--native_bind_coef", type=float, default=0.2)
    parser.add_argument("--num_basis", type=int, default=16)
    parser.add_argument("--prompt_length", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory exists and is not empty: {output}"
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

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
    opt.v_feat_dirs = [
        str(ROOT / "Soccergmr/clip"),
        str(ROOT / "Soccergmr/slowfast"),
    ]
    opt.mr_only = False
    opt.lw_saliency = 1
    opt.use_exist_head = True
    opt.exist_loss_coef = 1.0
    opt.exist_gate_thd = 0.3
    opt.exist_pool = "max"
    opt.query_cgp_use_semantic_mask = True
    opt.component_variant = "ls_dq_cgp_delta_zero_train"

    metadata = {
        "variant": opt.component_variant,
        "intervention": "e_adapt = frf_norm(e_static_expanded)",
        "training": "from_scratch",
        "seed": args.seed,
        "model": "StrictDeltaZero_LateSemantic_MomentDETR",
        "use_exist_head": True,
        "native_binding_coef": args.native_bind_coef,
        "num_basis": args.num_basis,
        "prompt_length": args.prompt_length,
        "query_cgp_use_semantic_mask": True,
        "epochs": args.epochs,
        "early_stop_patience": int(opt.max_es_cnt),
        "lr": args.lr,
    }
    (output / "experiment.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_dataset = StartEndDataset(
        **build_dataset_config(
            opt, opt.train_path, load_labels=True, keep_empty_gt=True
        )
    )
    val_dataset = StartEndDataset(
        **build_dataset_config(
            opt, opt.eval_path, load_labels=True, keep_empty_gt=False
        )
    )

    base_model, criterion = build_base_model(opt)
    model = StrictDeltaZeroLSDQCGPModel(
        base_model=base_model,
        num_basis=args.num_basis,
        prompt_length=args.prompt_length,
        router_hidden_dim=256,
        frf_hidden_dim=512,
        temperature=1.0,
    )
    install_ls_dq_cgp_loss(
        criterion, model, coefficient=args.native_bind_coef
    )
    model.to(opt.device)
    criterion.to(opt.device)

    # Residual-generation parameters are intentionally retained for exact
    # architecture/checkpoint compatibility.  Because Delta E_q is clamped to
    # zero, they receive no prediction-loss gradient and remain unchanged.
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=opt.lr,
        weight_decay=opt.wd,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, opt.lr_drop
    )

    logger.info(
        "Training strict Delta E_q=0 control: seed=%d, exist_head=True, epochs=%d",
        args.seed,
        args.epochs,
    )
    train(
        model,
        criterion,
        optimizer,
        scheduler,
        train_dataset,
        val_dataset,
        opt,
    )


if __name__ == "__main__":
    main()

