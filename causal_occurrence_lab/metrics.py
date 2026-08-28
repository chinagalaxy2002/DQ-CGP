"""Metrics used by the causal occurrence-binding experiments.

The implementation deliberately lives outside the released analysis harness.
In particular, duplicate attribution is one-to-one per prediction, so the
reported duplicate-attribution rate cannot become negative.
"""

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


def iou_matrix(
    predictions: Sequence[Sequence[float]],
    gt_windows: Sequence[Sequence[float]],
) -> np.ndarray:
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
    """Coverage, one-to-one DAR, and pairwise prediction overlap.

    For each top-k prediction ``p_j``, its sole attribution is
    ``argmax_k IoU(p_j, g_k)``.  A prediction is a valid hit only when that
    maximum is at least the requested threshold.  This is the corrected DAR
    definition from the experiment protocol.
    """

    matrix = iou_matrix(predictions, gt_windows)
    result: dict[str, Any] = {}
    for k in ks:
        top_matrix = matrix[: min(int(k), len(predictions))]
        pairwise = 0.0
        if top_matrix.shape[0] >= 2:
            pairwise = float(
                np.mean(
                    [
                        temporal_iou(predictions[i], predictions[j])
                        for i in range(top_matrix.shape[0])
                        for j in range(i + 1, top_matrix.shape[0])
                    ]
                )
            )
        result[f"pairwise_iou@{k}"] = pairwise

        for threshold in thresholds:
            key = f"{int(k)}_{int(round(float(threshold) * 10)):02d}"
            if not gt_windows:
                result[f"coverage@{key}"] = None
                result[f"duplicate_rate@{key}"] = None
                result[f"valid_hits@{key}"] = 0
                result[f"unique_attributed_gt@{key}"] = 0
                continue

            covered = (
                (top_matrix >= float(threshold)).any(axis=0)
                if top_matrix.size
                else np.zeros(len(gt_windows), dtype=bool)
            )
            if top_matrix.size:
                best_gt = top_matrix.argmax(axis=1)
                best_iou = top_matrix.max(axis=1)
                valid = best_iou >= float(threshold)
                unique_attributed = len(set(best_gt[valid].tolist()))
                valid_hits = int(valid.sum())
            else:
                unique_attributed = 0
                valid_hits = 0
            result[f"coverage@{key}"] = float(covered.sum() / len(gt_windows))
            result[f"valid_hits@{key}"] = valid_hits
            result[f"unique_attributed_gt@{key}"] = int(unique_attributed)
            result[f"duplicate_rate@{key}"] = (
                float(1.0 - unique_attributed / valid_hits) if valid_hits else None
            )
    return result


def attention_on_valid_video(
    attention: np.ndarray,
    valid_video_mask: Sequence[bool],
) -> np.ndarray:
    """Trim video attention to valid clips and renormalize each query."""

    values = np.asarray(attention, dtype=np.float64)
    if values.ndim == 3:  # [heads, queries, source]
        values = values.mean(axis=0)
    if values.ndim != 2:
        raise ValueError(f"expected [queries, source] attention, got {values.shape}")
    valid = np.asarray(valid_video_mask, dtype=bool)
    values = values[:, : len(valid)].copy()
    valid_length = int(valid.sum())
    values = values[:, :valid_length]
    values *= valid[:valid_length][None, :]
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


def _dominant_collision(details: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    denominator = 0
    numerator = 0
    for left in range(len(details)):
        for right in range(left + 1, len(details)):
            first = details[left]
            second = details[right]
            if first["gt_idx"] == second["gt_idx"]:
                continue
            denominator += 1
            numerator += int(first["dominant_gt"] == second["dominant_gt"])
    return numerator, denominator


def binding_metrics(
    attention: np.ndarray | None,
    gt_windows: Sequence[Sequence[float]],
    matched_query_indices: Sequence[int],
    matched_gt_indices: Sequence[int],
    *,
    clip_length: float,
    duration: float,
) -> dict[str, Any] | None:
    """Raw and length-normalized occurrence evidence metrics.

    ``R[j,k] = E[j,k] / (|G_k|/T + eps)`` uses the number of overlapping valid
    clips as ``|G_k|``.  This is the discrete feature-level analogue of the
    protocol's temporal-length normalization.
    """

    if attention is None or not gt_windows or not matched_query_indices:
        return None
    values = np.asarray(attention, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected [queries, clips] attention, got {values.shape}")
    masks = _clip_overlap_mask(gt_windows, values.shape[-1], clip_length, duration)
    fractions = masks.sum(axis=1).astype(np.float64) / max(values.shape[-1], 1)
    details = []
    eps = np.finfo(np.float64).eps
    for query_index, gt_index in zip(matched_query_indices, matched_gt_indices):
        query_attention = values[int(query_index)]
        masses = (query_attention[None, :] * masks).sum(axis=1)
        normalized_masses = masses / np.maximum(fractions, eps)
        own = int(gt_index)
        others = np.delete(masses, own)
        others_norm = np.delete(normalized_masses, own)
        other_mass = float(others.max()) if len(others) else None
        other_norm = float(others_norm.max()) if len(others_norm) else None
        details.append(
            {
                "query_idx": int(query_index),
                "gt_idx": own,
                "own_mass": float(masses[own]),
                "other_mass": other_mass,
                "binding_margin": float(masses[own] - other_mass) if other_mass is not None else None,
                "own_mass_norm": float(normalized_masses[own]),
                "other_mass_norm": other_norm,
                "binding_margin_norm": (
                    float(normalized_masses[own] - other_norm)
                    if other_norm is not None
                    else None
                ),
                "dominant_gt": int(np.argmax(masses)),
                "dominant_gt_norm": int(np.argmax(normalized_masses)),
                "evidence_mass": [float(value) for value in masses],
                "evidence_enrichment": [float(value) for value in normalized_masses],
                "gt_clip_fraction": [float(value) for value in fractions],
            }
        )

    margins = [item["binding_margin"] for item in details if item["binding_margin"] is not None]
    margins_norm = [item["binding_margin_norm"] for item in details if item["binding_margin_norm"] is not None]
    own_masses = [item["own_mass"] for item in details]
    own_masses_norm = [item["own_mass_norm"] for item in details]
    aec = float(np.mean([item["dominant_gt"] == item["gt_idx"] for item in details]))
    aec_norm = float(np.mean([item["dominant_gt_norm"] == item["gt_idx"] for item in details]))
    collision_num, collision_den = _dominant_collision(details)
    normalized_details = [
        {**item, "dominant_gt": item["dominant_gt_norm"]} for item in details
    ]
    collision_num_norm, _ = _dominant_collision(normalized_details)
    return {
        "aec": aec,
        "aec_norm": aec_norm,
        "binding_margin": float(np.mean(margins)) if margins else None,
        "binding_margin_norm": float(np.mean(margins_norm)) if margins_norm else None,
        "own_mass": float(np.mean(own_masses)) if own_masses else None,
        "own_mass_norm": float(np.mean(own_masses_norm)) if own_masses_norm else None,
        "ecr": collision_num / collision_den if collision_den else None,
        "ecr_norm": collision_num_norm / collision_den if collision_den else None,
        "collision_pairs": collision_den,
        "matched": details,
    }


def route_metrics(
    basis_weights: np.ndarray | None,
    query_indices: Sequence[int] | None = None,
) -> dict[str, Any] | None:
    """Conditional and marginal basis-routing statistics."""

    if basis_weights is None:
        return None
    weights = np.asarray(basis_weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] == 0:
        return None
    if query_indices is not None:
        if not query_indices:
            return None
        weights = weights[list(query_indices)]
    eps = np.finfo(np.float64).eps
    entropy = -(weights * np.log(np.maximum(weights, eps))).sum(axis=1)
    normalized = entropy / math.log(weights.shape[1]) if weights.shape[1] > 1 else entropy * 0.0
    marginal = weights.mean(axis=0)
    marginal_entropy = float(-(marginal * np.log(np.maximum(marginal, eps))).sum())
    marginal_entropy_norm = marginal_entropy / math.log(weights.shape[1]) if weights.shape[1] > 1 else 0.0
    return {
        "route_entropy": float(normalized.mean()),
        "route_entropy_std": float(normalized.std()),
        "marginal_entropy": marginal_entropy,
        "marginal_entropy_norm": marginal_entropy_norm,
        "effective_basis_count": float(np.exp(marginal_entropy)),
        "marginal_usage": [float(value) for value in marginal],
        "argmax_usage": np.bincount(weights.argmax(axis=1), minlength=weights.shape[1]).astype(int).tolist(),
        "num_queries": int(weights.shape[0]),
    }


def clean_multi_occurrence(gt_windows: Sequence[Sequence[float]], threshold: float = 0.1) -> bool:
    """Return whether all GT occurrences are non-overlapping at the threshold."""

    if len(gt_windows) < 2:
        return False
    return all(
        temporal_iou(gt_windows[left], gt_windows[right]) < float(threshold)
        for left in range(len(gt_windows))
        for right in range(left + 1, len(gt_windows))
    )


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value
