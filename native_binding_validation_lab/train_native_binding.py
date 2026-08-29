"""Train pure Moment-DETR with native D1 matched binding supervision."""

from __future__ import annotations

import argparse
import json
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

from config import BaseOptions  # noqa: E402
from dataset import StartEndDataset  # noqa: E402
from evaluate import setup_model  # noqa: E402
from training.moment_detr_gmr.train import build_dataset_config, train  # noqa: E402

from native_binding_validation_lab.native_binding import (  # noqa: E402
    NativeD1AttentionCapture,
    install_native_binding_loss,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    manager = BaseOptions("moment_detr", "soccer_gmr", "clip_slowfast")
    manager.parse()
    opt = manager.option
    opt.seed = args.seed
    opt.device = args.device
    opt.results_dir = str(output)
    opt.ckpt_filepath = str(output / opt.ckpt_filename)
    opt.train_log_filepath = str(output / opt.train_log_filename)
    opt.eval_log_filepath = str(output / opt.eval_log_filename)
    opt.train_path = str(ROOT / "data/label/Standard/train.jsonl")
    opt.eval_path = str(ROOT / "data/label/Standard/val.jsonl")
    opt.t_feat_dir = str(ROOT / "Soccergmr/clip_text")
    opt.v_feat_dirs = [str(ROOT / "Soccergmr/clip"), str(ROOT / "Soccergmr/slowfast")]
    opt.mr_only = True
    opt.lw_saliency = 0
    opt.component_variant = "native_binding"
    opt.native_binding_coef = 0.2
    (output / "experiment.json").write_text(json.dumps({
        "variant": "native_binding", "seed": args.seed,
        "model": "plain_moment_detr", "native_binding_coef": 0.2,
        "extra_trainable_parameters": 0,
    }, indent=2) + "\n")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train_dataset = StartEndDataset(**build_dataset_config(
        opt, opt.train_path, load_labels=True, keep_empty_gt=True,
    ))
    val_dataset = StartEndDataset(**build_dataset_config(
        opt, opt.eval_path, load_labels=True, keep_empty_gt=False,
    ))
    model, criterion, optimizer, scheduler = setup_model(opt)
    capture = NativeD1AttentionCapture(model)
    install_native_binding_loss(criterion, capture, coefficient=0.2)
    train(model, criterion, optimizer, scheduler, train_dataset, val_dataset, opt)


if __name__ == "__main__":
    main()
