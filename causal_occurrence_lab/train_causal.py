"""Train controlled causal variants without modifying the release trainer.

The default protocol is the released one: seed 2023, AdamW, lr 5e-5,
batch-size 8, 400 epochs, patience 50, and validation MR-full-mAP checkpoint
selection.  Variant-specific changes are confined to runtime controls in this
directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pprint
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from causal_occurrence_lab.common import REPO_ROOT
from causal_occurrence_lab.controls import (
    install_criterion_controls,
    install_native_binding_capture,
    install_query_cgp_controls,
)

# ``training.moment_detr_gmr.evaluate`` retains the release code's historical
# top-level imports (config.py/dataset.py), so expose that directory too.
TRAIN_ROOT = REPO_ROOT / "training" / "moment_detr_gmr"
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

from config import BaseOptions
from dataset import StartEndDataset, prepare_batch_inputs, start_end_collate
from evaluate import eval_epoch, setup_model
from models.moment_detr_gmr.utils.basic_utils import AverageMeter, rename_latest_to_best, save_checkpoint, write_log
from models.moment_detr_gmr.utils.model_utils import count_parameters
from experiments.vmr_cgp.query_checkpoint import load_query_cgp_state_compat
from experiments.temporal_cgp.checkpoint import load_model_state_compat

LOGGER = logging.getLogger("causal_occurrence_lab.train")
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {"dq": False, "bind": 0.0, "route": 0.0, "inject": False, "target": "matched"},
    "full": {"dq": True, "bind": 0.2, "route": 0.01, "inject": True, "target": "matched"},
    # Independent from the released checkpoint: this alias makes the exact
    # seed-2023 causal-harness reproduction explicit in run directories.
    "full_repro": {"dq": True, "bind": 0.2, "route": 0.01, "inject": True, "target": "matched"},
    "no_bind": {"dq": True, "bind": 0.0, "route": 0.01, "inject": True, "target": "matched"},
    "supervision_only": {"dq": True, "bind": 0.2, "route": 0.0, "inject": False, "target": "matched"},
    "union_bind": {"dq": True, "bind": 0.2, "route": 0.01, "inject": True, "target": "union"},
    "wrong_bind": {"dq": True, "bind": 0.2, "route": 0.01, "inject": True, "target": "rolled"},
    "no_route": {"dq": True, "bind": 0.2, "route": 0.0, "inject": True, "target": "matched"},
    "architecture_only": {"dq": True, "bind": 0.0, "route": 0.0, "inject": True, "target": "matched"},
    "native_bind": {"dq": False, "bind": 0.2, "route": 0.0, "inject": False, "target": "matched"},
}


def set_seed(seed: int, use_cuda: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def dataset_config(opt: Any, data_path: str, *, keep_empty_gt: bool) -> EasyDict:
    return EasyDict(
        dset_name=opt.dset_name,
        domain=None,
        data_path=data_path,
        ctx_mode=opt.ctx_mode,
        v_feat_dirs=opt.v_feat_dirs,
        a_feat_dirs=None,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        v_feat_types=opt.v_feat_types,
        a_feat_types=None,
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        max_a_l=opt.max_a_l,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        load_labels=True,
        mr_only=True,
        keep_empty_gt=keep_empty_gt,
        use_query_attention_mask=bool(getattr(opt, "use_query_attention_mask", False)),
        return_query_semantic_mask=bool(getattr(opt, "query_cgp_use_semantic_mask", False)),
    )


def configure_options(args: argparse.Namespace) -> Any:
    spec = VARIANTS[args.variant]
    model_name = "moment_detr_vmr_cgp_v3" if spec["dq"] else "moment_detr"
    manager = BaseOptions(model_name, "soccer_gmr", "clip_slowfast", args.resume)
    manager.parse()
    opt = manager.option
    opt.seed = args.seed
    opt.device = args.device
    opt.train_path = args.train_path or str(REPO_ROOT / "data" / "label" / "Standard" / "train.jsonl")
    opt.eval_path = args.eval_path or str(REPO_ROOT / "data" / "label" / "Standard" / "val.jsonl")
    opt.t_feat_dir = args.text_features
    opt.v_feat_dirs = list(args.video_features)
    opt.use_query_cgp = bool(spec["dq"])
    opt.use_tcgp = False
    opt.use_vmr_cgp = False
    opt.query_cgp_binding_loss_coef = float(spec["bind"])
    opt.query_cgp_route_loss_coef = float(spec["route"])
    opt.query_cgp_binding_target = str(spec["target"])
    opt.query_cgp_inject = bool(spec["inject"])
    if args.query_cgp_binding_loss_coef is not None:
        opt.query_cgp_binding_loss_coef = args.query_cgp_binding_loss_coef
    if args.query_cgp_route_loss_coef is not None:
        opt.query_cgp_route_loss_coef = args.query_cgp_route_loss_coef
    if args.query_cgp_binding_target is not None:
        opt.query_cgp_binding_target = args.query_cgp_binding_target
    if args.query_cgp_inject is not None:
        opt.query_cgp_inject = args.query_cgp_inject

    opt.lr = args.lr if args.lr is not None else opt.lr
    opt.n_epoch = args.n_epoch if args.n_epoch is not None else opt.n_epoch
    opt.bsz = args.bsz if args.bsz is not None else opt.bsz
    opt.eval_bsz = args.eval_bsz if args.eval_bsz is not None else opt.eval_bsz
    opt.max_es_cnt = args.max_es_cnt if args.max_es_cnt is not None else opt.max_es_cnt
    if args.results_dir:
        opt.results_dir = str(Path(args.results_dir).resolve())
    else:
        opt.results_dir = str(
            REPO_ROOT / "outputs" / "causal_ablation" / f"{args.variant}_seed{args.seed}"
        )
    opt.ckpt_filepath = str(Path(opt.results_dir) / opt.ckpt_filename)
    opt.train_log_filepath = str(Path(opt.results_dir) / opt.train_log_filename)
    opt.eval_log_filepath = str(Path(opt.results_dir) / opt.eval_log_filename)
    opt.mr_only = True
    opt.lw_saliency = 0
    return opt


def configure_controls(model: Any, criterion: Any, args: argparse.Namespace) -> None:
    spec = VARIANTS[args.variant]
    if args.variant == "baseline":
        return
    if spec["dq"]:
        install_query_cgp_controls(model, inject_residual=bool(args.query_cgp_inject))
        install_criterion_controls(
            criterion,
            binding_target=str(args.query_cgp_binding_target),
        )
        criterion.weight_dict["loss_query_cgp_bind"] = float(args.query_cgp_binding_loss_coef)
        criterion.weight_dict["loss_query_cgp_route"] = float(args.query_cgp_route_loss_coef)
    else:
        install_native_binding_capture(model)
        if "native_bind" not in criterion.losses:
            criterion.losses.append("native_bind")
        criterion.weight_dict["loss_native_bind"] = float(args.query_cgp_binding_loss_coef)
        install_criterion_controls(
            criterion,
            binding_target=str(args.query_cgp_binding_target),
            native_binding=True,
        )


def load_resume(model: Any, args: argparse.Namespace) -> None:
    if not args.resume:
        return
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    if VARIANTS[args.variant]["dq"]:
        load_query_cgp_state_compat(
            model, checkpoint["model"], allow_initialize_query_cgp=True
        )
    else:
        load_model_state_compat(model, checkpoint["model"])
    LOGGER.info("loaded model warm start: %s", args.resume)


def train_epoch(model: Any, criterion: Any, loader: DataLoader, optimizer: Any, opt: Any, epoch: int) -> dict[str, float]:
    model.train()
    criterion.train()
    meters: defaultdict[str, AverageMeter] = defaultdict(AverageMeter)
    for batch_index, batch in enumerate(tqdm(loader, desc=f"train epoch {epoch + 1}")):
        if opt.max_train_batches is not None and batch_index >= opt.max_train_batches:
            break
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device)
        outputs = model(**model_inputs)
        loss_dict = criterion(outputs, targets)
        losses = sum(
            loss_dict[key] * criterion.weight_dict[key]
            for key in loss_dict
            if key in criterion.weight_dict
        )
        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        loss_dict["loss_overall"] = float(losses.detach())
        for key, value in loss_dict.items():
            value_float = float(value.detach()) if torch.is_tensor(value) else float(value)
            weighted = value_float * criterion.weight_dict[key] if key in criterion.weight_dict else value_float
            meters[key].update(weighted)
    write_log(opt, epoch, meters)
    return {key: float(value.avg) for key, value in meters.items()}


def save_trajectory_snapshot(model: Any, optimizer: Any, scheduler: Any, epoch: int, opt: Any, trajectory_epochs: set[int]) -> None:
    if epoch + 1 not in trajectory_epochs:
        return
    path = Path(opt.results_dir) / "trajectory" / f"epoch_{epoch + 1:03d}.ckpt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "opt": opt,
        },
        path,
    )
    LOGGER.info("saved trajectory snapshot: %s", path)


def train(model: Any, criterion: Any, optimizer: Any, scheduler: Any, train_dataset: Any, val_dataset: Any, opt: Any, trajectory_epochs: set[int]) -> None:
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"
    loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
    )
    best_score = -float("inf")
    es_count = 0
    submission_filename = f"latest_{opt.dset_name}_val_preds.jsonl"
    trajectory_log = Path(opt.results_dir) / "trajectory_losses.jsonl"
    for epoch in trange(opt.n_epoch, desc=f"{opt.causal_variant} epochs"):
        loss_summary = train_epoch(model, criterion, loader, optimizer, opt, epoch)
        with trajectory_log.open("a", encoding="utf-8") as handle:
            import json
            handle.write(json.dumps({"epoch": epoch + 1, **loss_summary}) + "\n")
        scheduler.step()
        save_trajectory_snapshot(model, optimizer, scheduler, epoch, opt, trajectory_epochs)
        if (epoch + 1) % opt.eval_epoch_interval != 0:
            continue
        with torch.no_grad():
            metrics, eval_loss, latest_paths = eval_epoch(
                epoch + 1, model, val_dataset, opt, submission_filename, criterion
            )
        write_log(opt, epoch + 1, eval_loss, metrics=metrics, mode="val")
        brief = metrics["brief"] if metrics is not None else {}
        LOGGER.info("epoch=%d train=%s val=%s", epoch + 1, pprint.pformat(loss_summary), pprint.pformat(brief))
        score = float(brief.get("MR-full-mAP", -float("inf")))
        if score > best_score:
            best_score = score
            es_count = 0
            save_checkpoint(model, optimizer, scheduler, epoch, opt)
            rename_latest_to_best(latest_paths)
            LOGGER.info("updated best checkpoint: MR-full-mAP=%.4f", score)
        else:
            es_count += 1
            if es_count >= int(opt.max_es_cnt):
                LOGGER.info("early stopping at epoch %d; best MR-full-mAP=%.4f", epoch + 1, best_score)
                break


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n-epoch", "--n_epoch", dest="n_epoch", type=int, default=None)
    parser.add_argument("--bsz", type=int, default=None)
    parser.add_argument("--eval-bsz", "--eval_bsz", dest="eval_bsz", type=int, default=None)
    parser.add_argument("--max-es-cnt", "--max_es_cnt", dest="max_es_cnt", type=int, default=None)
    parser.add_argument("--train-path", "--train_path", dest="train_path", default=None)
    parser.add_argument("--eval-path", "--eval_path", dest="eval_path", default=None)
    parser.add_argument("--text-features", "--t_feat_dir", dest="text_features", default=str(REPO_ROOT / "Soccergmr" / "clip_text"))
    parser.add_argument("--video-features", "--v_feat_dirs", dest="video_features", nargs=2, default=[str(REPO_ROOT / "Soccergmr" / "clip"), str(REPO_ROOT / "Soccergmr" / "slowfast")])
    parser.add_argument("--results-dir", "--results_dir", dest="results_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query-cgp-binding-loss-coef", "--query_cgp_binding_loss_coef", dest="query_cgp_binding_loss_coef", type=float, default=None)
    parser.add_argument("--query-cgp-route-loss-coef", "--query_cgp_route_loss_coef", dest="query_cgp_route_loss_coef", type=float, default=None)
    parser.add_argument("--query-cgp-binding-target", "--query_cgp_binding_target", dest="query_cgp_binding_target", choices=["matched", "union", "rolled"], default=None)
    parser.add_argument("--query-cgp-inject", "--query_cgp_inject", dest="query_cgp_inject", action="store_true", default=None)
    parser.add_argument("--no-query-cgp-inject", "--no_query_cgp_inject", dest="query_cgp_inject", action="store_false")
    parser.add_argument("--trajectory-epochs", default="1,5,10,20,40,80,best", help="Snapshot epochs; 'best' is represented by the normal best checkpoint.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional smoke-test cap; omitted in the paper protocol.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    spec = VARIANTS[args.variant]
    if args.query_cgp_binding_loss_coef is None:
        args.query_cgp_binding_loss_coef = float(spec["bind"])
    if args.query_cgp_route_loss_coef is None:
        args.query_cgp_route_loss_coef = float(spec["route"])
    if args.query_cgp_binding_target is None:
        args.query_cgp_binding_target = str(spec["target"])
    if args.query_cgp_inject is None:
        args.query_cgp_inject = bool(spec["inject"])
    opt = configure_options(args)
    opt.causal_variant = args.variant
    opt.max_train_batches = args.max_train_batches
    Path(opt.results_dir).mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        # The exact run directory is user-selected by --results-dir/default
        # variant path; remove only its known files through the release helper.
        import shutil
        shutil.rmtree(opt.results_dir)
        Path(opt.results_dir).mkdir(parents=True, exist_ok=True)
    (Path(opt.results_dir) / "variant.json").write_text(
        json.dumps({
            "variant": args.variant,
            "seed": args.seed,
            "binding_loss_coef": args.query_cgp_binding_loss_coef,
            "route_loss_coef": args.query_cgp_route_loss_coef,
            "binding_target": args.query_cgp_binding_target,
            "inject_residual": args.query_cgp_inject,
            "lr": opt.lr,
            "batch_size": opt.bsz,
            "epochs": opt.n_epoch,
            "checkpoint_selection": "val MR-full-mAP",
        }, indent=2),
        encoding="utf-8",
    )
    set_seed(opt.seed, use_cuda=opt.device.startswith("cuda"))
    train_dataset = StartEndDataset(**dataset_config(opt, opt.train_path, keep_empty_gt=bool(getattr(opt, "use_exist_head", False))))
    val_dataset = StartEndDataset(**dataset_config(opt, opt.eval_path, keep_empty_gt=False))
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("empty train/validation dataset; check feature directories and labels")
    model, criterion, optimizer, scheduler = setup_model(opt)
    load_resume(model, args)
    configure_controls(model, criterion, args)
    count_parameters(model)
    trajectory_epochs = {
        int(item.strip()) for item in args.trajectory_epochs.split(",")
        if item.strip().isdigit()
    }
    train(model, criterion, optimizer, scheduler, train_dataset, val_dataset, opt, trajectory_epochs)


if __name__ == "__main__":
    main()
