"""Fair Native Binding + Exist control aligned with LS-DQ-CGP+Exist."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "training" / "moment_detr_gmr"
for path in (ROOT, TRAIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import BaseOptions
from dataset import StartEndDataset
from evaluate import setup_model
from training.moment_detr_gmr.train import build_dataset_config, train
from native_binding_validation_lab.native_binding import (
    NativeD1AttentionCapture,
    install_native_binding_loss,
)
from ls_dq_cgp_ablation_lab.common import configure_standard_opt, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
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
    opt.query_cgp_use_semantic_mask = False
    opt.component_variant = "native_binding_exist_aligned"
    opt.native_binding_coef = args.native_bind_coef
    metadata = {
        "variant": opt.component_variant,
        "protocol": "from_scratch",
        "model": "plain_moment_detr",
        "seed": args.seed,
        "use_exist_head": True,
        "mr_only": False,
        "lw_saliency": 1,
        "native_binding_coef": args.native_bind_coef,
        "extra_trainable_parameters": 0,
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
    model, criterion, optimizer, scheduler = setup_model(opt)
    capture = NativeD1AttentionCapture(model)
    install_native_binding_loss(
        criterion, capture, coefficient=args.native_bind_coef
    )
    train(
        model, criterion, optimizer, scheduler,
        train_dataset, val_dataset, opt,
    )


if __name__ == "__main__":
    main()

