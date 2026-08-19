from __future__ import annotations

import argparse
import logging
import os
import pprint
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import BaseOptions
from dataset import StartEndDataset, prepare_batch_inputs, start_end_collate
from evaluate import eval_epoch, setup_model
from models.moment_detr_gmr.utils.basic_utils import (
    AverageMeter,
    rename_latest_to_best,
    save_checkpoint,
    write_log,
)
from models.moment_detr_gmr.utils.model_utils import count_parameters, ModelEMA
from experiments.temporal_cgp.checkpoint import load_model_state_compat
from experiments.vmr_cgp.checkpoint import load_vmr_cgp_state_compat
from experiments.vmr_cgp.query_checkpoint import load_query_cgp_state_compat

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i):
    logger.info("[Epoch %d]", epoch_i + 1)
    model.train()
    criterion.train()
    loss_meters = defaultdict(AverageMeter)

    for batch in tqdm(train_loader, desc="Training Iteration"):
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device)
        outputs = model(**model_inputs)
        loss_dict = criterion(outputs, targets)
        losses = sum(
            loss_dict[k] * criterion.weight_dict[k]
            for k in loss_dict.keys()
            if k in criterion.weight_dict
        )

        optimizer.zero_grad()
        losses.backward()
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()

        loss_dict["loss_overall"] = float(losses)
        for k, v in loss_dict.items():
            loss_meters[k].update(float(v) * criterion.weight_dict[k] if k in criterion.weight_dict else float(v))

    write_log(opt, epoch_i, loss_meters)

def train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
    )

    model_ema = None
    if opt.model_ema:
        logger.info("Using model EMA")
        model_ema = ModelEMA(model, decay=opt.ema_decay)

    prev_best_score = 0
    es_cnt = 0
    save_submission_filename = f"latest_{opt.dset_name}_val_preds.jsonl"

    for epoch_i in trange(opt.n_epoch, desc="Epoch"):
        train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i)
        lr_scheduler.step()

        if model_ema is not None:
            model_ema.update(model)

        if (epoch_i + 1) % opt.eval_epoch_interval != 0:
            continue

        with torch.no_grad():
            eval_model = model_ema.module if model_ema is not None else model
            metrics, eval_loss_meters, latest_file_paths = eval_epoch(
                epoch_i,
                eval_model,
                val_dataset,
                opt,
                save_submission_filename,
                criterion,
            )

        write_log(opt, epoch_i, eval_loss_meters, metrics=metrics, mode="val")
        logger.info("metrics %s", pprint.pformat(metrics["brief"], indent=4))
        stop_score = metrics["brief"].get("MR-full-mAP", 0)

        if stop_score > prev_best_score:
            prev_best_score = stop_score
            save_checkpoint(model, optimizer, lr_scheduler, epoch_i, opt)
            rename_latest_to_best(latest_file_paths)
            es_cnt = 0
            logger.info("Updated best checkpoint.")
        else:
            es_cnt += 1
            logger.info("Early stop counter: %d/%d", es_cnt, opt.max_es_cnt)
            if es_cnt >= int(opt.max_es_cnt):
                logger.info("Early stopping at epoch %d. Best score %.4f", epoch_i + 1, prev_best_score)
                break

def build_dataset_config(opt, data_path, load_labels=True, keep_empty_gt=False):
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
        load_labels=load_labels,
        mr_only=bool(getattr(opt, "mr_only", True)),
        keep_empty_gt=keep_empty_gt,
        use_query_attention_mask=bool(getattr(opt, "use_query_attention_mask", False)),
        return_query_semantic_mask=bool(
            getattr(opt, "query_cgp_use_semantic_mask", False)
        ),
    )

def main(opt, resume=None):
    logger.info("Setup config, data and model...")
    set_seed(opt.seed, use_cuda=opt.device == "cuda")

    train_dataset = StartEndDataset(**build_dataset_config(
        opt,
        opt.train_path,
        load_labels=True,
        keep_empty_gt=bool(getattr(opt, "use_exist_head", False)),
    ))
    val_dataset = StartEndDataset(**build_dataset_config(
        opt,
        opt.eval_path,
        load_labels=True,
        keep_empty_gt=False,
    ))

    if len(train_dataset) == 0:
        raise RuntimeError(
            "No usable training samples remain after feature validation. "
            f"Expected video features in {opt.v_feat_dirs} and text features in "
            f"{opt.t_feat_dir}. Check the feature extraction/download step."
        )
    if len(val_dataset) == 0:
        raise RuntimeError(
            "No usable validation samples remain after feature validation. "
            f"Expected video features in {opt.v_feat_dirs} and text features in "
            f"{opt.t_feat_dir}. Check the feature extraction/download step."
        )

    model, criterion, optimizer, lr_scheduler = setup_model(opt)
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if bool(getattr(opt, "use_query_cgp", False)):
            load_query_cgp_state_compat(
                model,
                checkpoint["model"],
                allow_initialize_query_cgp=True,
            )
        elif bool(getattr(opt, "use_vmr_cgp", False)):
            load_vmr_cgp_state_compat(
                model,
                checkpoint["model"],
                allow_initialize_vmr_cgp=True,
            )
        else:
            load_model_state_compat(
                model,
                checkpoint["model"],
                allow_initialize_tcgp=bool(getattr(opt, "use_tcgp", False)),
            )
        logger.info("Loaded model checkpoint: %s", resume)

    count_parameters(model)
    logger.info("Start training")
    train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Moment-DETR-GMR on Soccer-GMR features.")
    parser.add_argument(
        "--model", "-m", default="moment_detr",
        choices=[
            "moment_detr",
            "moment_detr_masked",
            "moment_detr_tcgp",
            "moment_detr_vmr_cgp",
            "moment_detr_vmr_cgp_v2",
            "moment_detr_vmr_cgp_v3",
        ],
    )
    parser.add_argument("--dataset", "-d", default="soccer_gmr", choices=["soccer_gmr"])
    parser.add_argument("--feature", "-f", default="clip_slowfast", choices=["clip_slowfast"])
    parser.add_argument("--resume", "-r", type=str, default=None, help="Optional checkpoint for fine-tuning.")
    parser.add_argument("--run_tag", type=str, default=None, help="Append a tag to the output directory.")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n_epoch", type=int, default=None)
    parser.add_argument("--bsz", type=int, default=None)
    parser.add_argument("--eval_bsz", type=int, default=None)
    parser.add_argument("--max_es_cnt", type=int, default=None)
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--eval_path", type=str, default=None)
    parser.add_argument("--t_feat_dir", type=str, default=None)
    parser.add_argument("--v_feat_dirs", type=str, nargs="+", default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Remove the output directory before training.")
    parser.add_argument("--mr_only", action="store_true", default=True, help="Disable saliency labels.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    option_manager = BaseOptions(args.model, args.dataset, args.feature, args.resume)
    option_manager.parse()
    opt = option_manager.option

    if args.run_tag:
        opt.results_dir = os.path.join(opt.results_dir, args.run_tag)
        opt.ckpt_filepath = os.path.join(opt.results_dir, opt.ckpt_filename)
        opt.train_log_filepath = os.path.join(opt.results_dir, opt.train_log_filename)
        opt.eval_log_filepath = os.path.join(opt.results_dir, opt.eval_log_filename)
    for name in ["lr", "n_epoch", "bsz", "eval_bsz", "max_es_cnt", "train_path", "eval_path", "t_feat_dir", "results_dir", "device"]:
        value = getattr(args, name)
        if value is not None:
            setattr(opt, name, value)
    if args.v_feat_dirs is not None:
        opt.v_feat_dirs = args.v_feat_dirs
    if args.results_dir is not None:
        opt.ckpt_filepath = os.path.join(opt.results_dir, opt.ckpt_filename)
        opt.train_log_filepath = os.path.join(opt.results_dir, opt.train_log_filename)
        opt.eval_log_filepath = os.path.join(opt.results_dir, opt.eval_log_filename)
    opt.mr_only = True
    opt.lw_saliency = 0

    option_manager.clean_and_makedirs(overwrite=args.overwrite)
    main(opt, resume=args.resume)
