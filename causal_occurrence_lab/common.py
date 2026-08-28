"""Shared runtime utilities for isolated causal experiments."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict

LAB_ROOT = Path(__file__).resolve().parent
REPO_ROOT = LAB_ROOT.parent
TRAIN_ROOT = REPO_ROOT / "training" / "moment_detr_gmr"
for path in (REPO_ROOT, TRAIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.temporal_cgp.checkpoint import load_model_state_compat
from causal_occurrence_lab.controls import install_query_cgp_controls
from experiments.vmr_cgp.query_ablation import apply_query_cgp_ablation
from experiments.vmr_cgp.query_checkpoint import (
    load_query_cgp_state_compat,
    restore_query_cgp_options,
)
from models.moment_detr_gmr.moment_detr import build_model
from models.moment_detr_gmr.utils.span_utils import span_cxw_to_xx
from training.moment_detr_gmr.config import BaseOptions
from training.moment_detr_gmr.dataset import (
    StartEndDataset,
    prepare_batch_inputs,
    start_end_collate,
)

LOGGER = logging.getLogger("causal_occurrence_lab")


def load_checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or "model" not in checkpoint:
        raise ValueError(f"checkpoint has no model state: {path}")
    return dict(checkpoint)


def _checkpoint_opt(checkpoint: Mapping[str, Any], key: str, default: Any = None) -> Any:
    saved = checkpoint.get("opt")
    if saved is None:
        return default
    if isinstance(saved, Mapping):
        return saved.get(key, default)
    return getattr(saved, key, default)


def configure_options(
    checkpoint: Mapping[str, Any],
    *,
    model_name: str,
    split: str,
    eval_path: str,
    text_features: str,
    video_features: Sequence[str],
    device: str,
    force_query_cgp: bool | None = None,
) -> Any:
    manager = BaseOptions(model_name, "soccer_gmr", "clip_slowfast")
    manager.parse()
    opt = manager.option
    opt.eval_path = eval_path
    opt.eval_split_name = split
    opt.t_feat_dir = text_features
    opt.v_feat_dirs = list(video_features)
    opt.device = device

    has_query = any(str(key).startswith("query_cgp.") for key in checkpoint["model"])
    if has_query:
        restore_query_cgp_options(opt, checkpoint)
    if force_query_cgp is not None:
        opt.use_query_cgp = bool(force_query_cgp)
    else:
        opt.use_query_cgp = bool(has_query)
    opt.use_tcgp = False
    opt.use_vmr_cgp = False
    opt.use_exist_head = any("exist_head" in str(key) for key in checkpoint["model"])
    return opt


def load_model_for_analysis(
    checkpoint_path: str,
    *,
    mode: str,
    model_name: str,
    split: str,
    eval_path: str,
    text_features: str,
    video_features: Sequence[str],
    device: str,
) -> tuple[torch.nn.Module, torch.nn.Module, Any, dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path)
    has_query = any(str(key).startswith("query_cgp.") for key in checkpoint["model"])
    if mode.startswith("dq_") and not has_query:
        raise ValueError(f"{mode} requires a DQ-CGP checkpoint: {checkpoint_path}")
    if mode == "dq_stripped" and not has_query:
        raise ValueError("dq_stripped requires query_cgp.* parameters to strip")

    is_stripped = mode == "dq_stripped"
    use_query = has_query and not is_stripped
    effective_model_name = "moment_detr_vmr_cgp_v3" if use_query else "moment_detr"
    opt = configure_options(
        checkpoint,
        model_name=effective_model_name if model_name == "auto" else model_name,
        split=split,
        eval_path=eval_path,
        text_features=text_features,
        video_features=video_features,
        device=device,
        force_query_cgp=use_query,
    )
    model, criterion = build_model(opt)
    state = checkpoint["model"]
    if is_stripped:
        state = {key: value for key, value in state.items() if not str(key).startswith("query_cgp.")}
    if use_query:
        load_query_cgp_state_compat(model, state)
    else:
        load_model_state_compat(model, state)
    model.to(device)
    criterion.to(device)
    model.eval()
    criterion.eval()

    if mode == "dq_beta_zero":
        apply_query_cgp_ablation(model, "beta_zero")
    elif mode not in {"baseline", "dq_active", "dq_stripped"}:
        raise ValueError(f"unsupported analysis mode: {mode}")
    # Training-only controls are persisted in checkpoints created by this
    # harness.  Reinstall the no-injection forward for SupervisionOnly so its
    # evaluation remains D1 -> D2, while still exposing binding diagnostics.
    saved_inject = _checkpoint_opt(checkpoint, "query_cgp_inject", None)
    if mode == "dq_active" and use_query and saved_inject is False:
        install_query_cgp_controls(model, inject_residual=False)
    return model, criterion, opt, checkpoint


def build_dataset_config(opt: Any, data_path: str) -> EasyDict:
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
        keep_empty_gt=True,
        use_query_attention_mask=bool(getattr(opt, "use_query_attention_mask", False)),
        return_query_semantic_mask=bool(getattr(opt, "query_cgp_use_semantic_mask", False)),
    )


def prediction_data(
    outputs: Mapping[str, torch.Tensor],
    metas: Sequence[Mapping[str, Any]],
) -> tuple[list[list[list[float]]], list[list[float]], list[list[int]]]:
    probabilities = F.softmax(outputs["pred_logits"], dim=-1)[..., 0].detach().cpu().numpy()
    spans = span_cxw_to_xx(outputs["pred_spans"]).detach().cpu().numpy()
    predictions, scores, indices = [], [], []
    for batch_index, meta in enumerate(metas):
        duration = float(meta["duration"])
        seconds = np.clip(spans[batch_index] * duration, 0.0, duration)
        order = np.argsort(-probabilities[batch_index], kind="stable")
        predictions.append([[float(seconds[i, 0]), float(seconds[i, 1])] for i in order])
        scores.append([float(probabilities[batch_index, i]) for i in order])
        indices.append([int(i) for i in order])
    return predictions, scores, indices


def submission_for_layer(
    records: Sequence[Mapping[str, Any]],
    layer: str,
) -> list[dict[str, Any]]:
    output = []
    for record in records:
        windows = record[f"{layer}_pred_windows"]
        scores = record[f"{layer}_scores"]
        item = {
            "qid": record["qid"],
            "query": record.get("query"),
            "vid": record.get("vid"),
            "pred_relevant_windows": [
                [float(window[0]), float(window[1]), float(score)]
                for window, score in zip(windows, scores)
            ],
        }
        if layer == "d2" and record.get("pred_exist_score") is not None:
            item["pred_exist_score"] = float(record["pred_exist_score"])
        else:
            item["pred_exist_score"] = float(max(scores)) if scores else 0.0
        output.append(item)
    return output


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def as_windows(value: Any) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    return [[float(item[0]), float(item[1])] for item in value if len(item) >= 2]


def mean_valid(values: Sequence[Any]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return float(np.mean(valid)) if valid else None
