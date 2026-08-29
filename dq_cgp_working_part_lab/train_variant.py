"""Train one DQ-CGP component ablation without editing production code."""

from __future__ import annotations

import argparse
import json
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

from config import BaseOptions  # noqa: E402
from dataset import StartEndDataset  # noqa: E402
from evaluate import setup_model  # noqa: E402
from training.moment_detr_gmr.train import build_dataset_config, train  # noqa: E402

from dq_cgp_working_part_lab.controls import install_residual_injection_control  # noqa: E402
from dq_cgp_working_part_lab.specs import VARIANTS, get_spec  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-epoch", type=int, default=400)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    spec = get_spec(args.variant)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Non-empty output directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    model_name = "moment_detr_vmr_cgp_v3" if spec["dq"] else "moment_detr"
    manager = BaseOptions(model_name, "soccer_gmr", "clip_slowfast")
    manager.parse()
    opt = manager.option
    opt.seed = args.seed
    opt.device = args.device
    opt.n_epoch = args.n_epoch
    opt.max_es_cnt = args.patience
    opt.results_dir = str(output)
    opt.ckpt_filepath = str(output / opt.ckpt_filename)
    opt.train_log_filepath = str(output / opt.train_log_filename)
    opt.eval_log_filepath = str(output / opt.eval_log_filename)
    opt.train_path = str(ROOT / "data" / "label" / "Standard" / "train.jsonl")
    opt.eval_path = str(ROOT / "data" / "label" / "Standard" / "val.jsonl")
    opt.t_feat_dir = str(ROOT / "Soccergmr" / "clip_text")
    opt.v_feat_dirs = [str(ROOT / "Soccergmr" / "clip"), str(ROOT / "Soccergmr" / "slowfast")]
    opt.mr_only = True
    opt.lw_saliency = 0
    opt.use_query_cgp = bool(spec["dq"])
    opt.use_tcgp = False
    opt.use_vmr_cgp = False
    opt.query_cgp_binding_loss_coef = float(spec["bind"])
    opt.query_cgp_route_loss_coef = float(spec["route"])
    opt.component_variant = args.variant
    opt.component_inject_residual = bool(spec["inject"])
    opt.max_train_batches = args.max_train_batches

    (output / "experiment.json").write_text(
        json.dumps({"variant": args.variant, "seed": args.seed, **spec}, indent=2) + "\n"
    )
    set_seed(args.seed)
    train_dataset = StartEndDataset(**build_dataset_config(
        opt, opt.train_path, load_labels=True,
        keep_empty_gt=bool(getattr(opt, "use_exist_head", False)),
    ))
    val_dataset = StartEndDataset(**build_dataset_config(
        opt, opt.eval_path, load_labels=True, keep_empty_gt=False,
    ))
    model, criterion, optimizer, scheduler = setup_model(opt)
    if spec["dq"]:
        criterion.weight_dict["loss_query_cgp_bind"] = float(spec["bind"])
        criterion.weight_dict["loss_query_cgp_route"] = float(spec["route"])
        install_residual_injection_control(model, bool(spec["inject"]))

    # The production trainer has no batch cap. A smoke-only cap is installed
    # by wrapping the dataset, never by changing production source.
    if args.max_train_batches is not None:
        limit = min(len(train_dataset), args.max_train_batches * opt.bsz)
        train_dataset.data = train_dataset.data[:limit]
    train(model, criterion, optimizer, scheduler, train_dataset, val_dataset, opt)


if __name__ == "__main__":
    main()
