"""Train an LS-DQ-CGP semantic component ablation from scratch."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
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
from models.moment_detr_gmr.moment_detr import build_model as build_base_model
from training.moment_detr_gmr.train import build_dataset_config, train
from ls_dq_cgp_lab.ls_dq_cgp_model import install_ls_dq_cgp_loss
from ls_dq_cgp_ablation_lab.ablation_model import (
    ABLATION_VARIANTS,
    AblationLSDQCGPModel,
)
from ls_dq_cgp_ablation_lab.common import configure_standard_opt, set_seed

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=ABLATION_VARIANTS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--native_bind_coef", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    manager = BaseOptions("moment_detr", "soccer_gmr", "clip_slowfast")
    manager.parse()
    opt = configure_standard_opt(
        manager.option, output, args.seed, args.device, args.lr, args.epochs
    )
    opt.query_cgp_use_semantic_mask = True
    opt.component_variant = f"ls_dq_cgp_ablation_{args.variant}"

    metadata = {
        "variant": args.variant,
        "protocol": "from_scratch",
        "seed": args.seed,
        "use_exist_head": True,
        "mr_only": False,
        "lw_saliency": 1,
        "native_binding_coef": args.native_bind_coef,
        "epochs": args.epochs,
        "early_stop_patience": int(opt.max_es_cnt),
        "lr": args.lr,
    }
    (output / "experiment.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    set_seed(args.seed)

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
    model = AblationLSDQCGPModel(
        base_model=base_model,
        num_basis=16,
        prompt_length=6,
        router_hidden_dim=256,
        frf_hidden_dim=512,
        temperature=1.0,
        variant=args.variant,
    )
    install_ls_dq_cgp_loss(
        criterion, model, coefficient=args.native_bind_coef
    )
    model.to(opt.device)
    criterion.to(opt.device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=opt.lr,
        weight_decay=opt.wd,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop)
    logger.info("Training semantic ablation %s from scratch", args.variant)
    train(
        model, criterion, optimizer, scheduler,
        train_dataset, val_dataset, opt,
    )


if __name__ == "__main__":
    main()

