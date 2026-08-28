"""Fixed-K occurrence coverage and evidence-binding metrics."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def temporal_iou(window_a: Sequence[float], window_b: Sequence[float]) -> float:
    start = max(float(window_a[0]), float(window_b[0]))
    end = min(float(window_a[1]), float(window_b[1]))
    intersection = max(0.0, end - start)
    union = max(float(window_a[1]), float(window_b[1])) - min(
        float(window_a[0]), float(window_b[0])
    )
    return intersection / union if union > 0.0 else 0.0


def _iou_matrix(predictions: Sequence[Sequence[float]], gt_windows: Sequence[Sequence[float]]) -> np.ndarray:
    if not predictions or not gt_windows:
        return np.zeros((len(predictions), len(gt_windows)), dtype=np.float64)
    return np.asarray(
        [[temporal_iou(prediction, gt) for gt in gt_windows] for prediction in predictions],
        dtype=np.float64,
    )


def fixed_k_metrics(
    predictions: Sequence[Sequence[float]],
    gt_windows: Sequence[Sequence[float]],
    ks: Iterable[int] = (1, 3, 5, 10),
    thresholds: Iterable[float] = (0.3, 0.5, 0.7),
) -> dict[str, Any]:
    """Compute fixed-K coverage, duplicate attribution, and pairwise overlap."""

    matrix = _iou_matrix(predictions, gt_windows)
    result: dict[str, Any] = {}
    for k in ks:
        top_matrix = matrix[: min(int(k), len(predictions))]
        pairwise = 0.0
        if top_matrix.shape[0] >= 2:
            prediction_ious = [
                temporal_iou(predictions[i], predictions[j])
                for i in range(top_matrix.shape[0])
                for j in range(i + 1, top_matrix.shape[0])
            ]
            pairwise = float(np.mean(prediction_ious))
        result[f"pairwise_iou@{k}"] = pairwise

        for threshold in thresholds:
            key = f"{int(k)}_{int(round(float(threshold) * 10)):02d}"
            if not gt_windows:
                result[f"coverage@{key}"] = None
                result[f"duplicate_rate@{key}"] = None
                result[f"valid_hits@{key}"] = 0
                continue

            covered = (
                (top_matrix >= float(threshold)).any(axis=0)
                if top_matrix.size
                else np.zeros(len(gt_windows), dtype=bool)
            )
            valid_prediction_hits = (
                (top_matrix.max(axis=1) >= float(threshold)).sum()
                if top_matrix.size
                else 0
            )
            unique_covered = int(covered.sum())
            valid_prediction_hits = int(valid_prediction_hits)
            result[f"coverage@{key}"] = unique_covered / len(gt_windows)
            result[f"valid_hits@{key}"] = valid_prediction_hits
            result[f"duplicate_rate@{key}"] = (
                1.0 - unique_covered / valid_prediction_hits
                if valid_prediction_hits
                else None
            )
    return result


def attention_on_valid_video(
    attention: np.ndarray,
    valid_video_mask: Sequence[bool],
) -> np.ndarray:
    """Trim native attention to video tokens and renormalize over valid clips."""

    values = np.asarray(attention, dtype=np.float64)
    if values.ndim == 3:  # [heads, queries, source]
        values = values.mean(axis=0)
    if values.ndim != 2:
        raise ValueError(f"expected [queries, source] attention, got {values.shape}")
    valid = np.asarray(valid_video_mask, dtype=bool)
    values = values[:, : len(valid)].copy()
    values *= valid[None, :]
    denominator = values.sum(axis=1, keepdims=True)
    return np.divide(values, np.maximum(denominator, np.finfo(np.float64).eps))


def _clip_overlap_mask(
    gt_windows: Sequence[Sequence[float]],
    num_clips: int,
    clip_length: float,
    duration: float,
) -> np.ndarray:
    clip_starts = np.arange(num_clips, dtype=np.float64) * float(clip_length)
    clip_ends = np.minimum(clip_starts + float(clip_length), float(duration))
    masks = []
    for gt in gt_windows:
        start = max(0.0, float(gt[0]))
        end = min(float(duration), float(gt[1]))
        overlap = (clip_starts < end) & (clip_ends > start)
        if not overlap.any() and num_clips:
            center = 0.5 * (clip_starts + clip_ends)
            overlap[int(np.argmin(np.abs(center - start)))] = True
        masks.append(overlap)
    return np.asarray(masks, dtype=bool)


def binding_metrics(
    attention: np.ndarray | None,
    gt_windows: Sequence[Sequence[float]],
    matched_query_indices: Sequence[int],
    matched_gt_indices: Sequence[int],
    *,
    clip_length: float,
    duration: float,
) -> dict[str, Any] | None:
    """Compute own mass, binding margin, AEC, and ECR for one layer."""

    if attention is None or not gt_windows or not matched_query_indices:
        return None
    masks = _clip_overlap_mask(
        gt_windows, attention.shape[-1], clip_length, duration
    )
    matched_details = []
    for query_index, gt_index in zip(matched_query_indices, matched_gt_indices):
        query_attention = attention[int(query_index)]
        masses = (query_attention[None, :] * masks).sum(axis=1)
        own_mass = float(masses[int(gt_index)])
        others = np.delete(masses, int(gt_index))
        other_mass = float(others.max()) if len(others) else None
        dominant_gt = int(np.argmax(masses)) if len(masses) else None
        matched_details.append(
            {
                "query_idx": int(query_index),
                "gt_idx": int(gt_index),
                "own_mass": own_mass,
                "other_mass": other_mass,
                "binding_margin": (
                    own_mass - other_mass if other_mass is not None else None
                ),
                "dominant_gt": dominant_gt,
                "evidence_mass": [float(value) for value in masses],
            }
        )

    margins = [
        item["binding_margin"]
        for item in matched_details
        if item["binding_margin"] is not None
    ]
    own_masses = [item["own_mass"] for item in matched_details]
    aec = np.mean(
        [item["dominant_gt"] == item["gt_idx"] for item in matched_details]
    )

    collision_denominator = 0
    collision_numerator = 0
    for left in range(len(matched_details)):
        for right in range(left + 1, len(matched_details)):
            first = matched_details[left]
            second = matched_details[right]
            if first["gt_idx"] == second["gt_idx"]:
                continue
            collision_denominator += 1
            collision_numerator += int(first["dominant_gt"] == second["dominant_gt"])

    return {
        "aec": float(aec),
        "binding_margin": float(np.mean(margins)) if margins else None,
        "own_mass": float(np.mean(own_masses)) if own_masses else None,
        "ecr": (
            collision_numerator / collision_denominator
            if collision_denominator
            else None
        ),
        "collision_pairs": collision_denominator,
        "matched": matched_details,
    }


def route_metrics(
    basis_weights: np.ndarray | None,
    matched_query_indices: Sequence[int],
) -> dict[str, float | None] | None:
    if basis_weights is None or not matched_query_indices:
        return None
    weights = np.asarray(basis_weights, dtype=np.float64)[list(matched_query_indices)]
    if weights.ndim != 2 or weights.shape[1] == 0:
        return None
    eps = np.finfo(np.float64).eps
    entropy = -(weights * np.log(np.maximum(weights, eps))).sum(axis=1)
    normalized = entropy / math.log(weights.shape[1]) if weights.shape[1] > 1 else entropy * 0.0
    return {
        "route_entropy": float(normalized.mean()),
        "route_entropy_std": float(normalized.std()),
    }


def jsonable(value: Any) -> Any:
    """Recursively convert NumPy scalar/array values for JSON output."""

    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value

