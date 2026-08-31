"""Shared configuration helpers for controlled ablations."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "training" / "moment_detr_gmr"


def configure_standard_opt(opt, output: Path, seed: int, device: str, lr: float, epochs: int):
    """Apply the fair LS-DQ-CGP+Exist training configuration."""
    opt.seed = seed
    opt.device = device
    opt.n_epoch = epochs
    opt.lr = lr
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
    opt.use_exist_head = True
    opt.exist_loss_coef = 1.0
    opt.exist_gate_thd = 0.3
    opt.exist_pool = "max"
    opt.mr_only = False
    opt.lw_saliency = 1
    return opt


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def portable_path(path: Path) -> str:
    try:
        return f"<repo>/{path.resolve().relative_to(ROOT)}"
    except ValueError:
        return str(path.resolve())

