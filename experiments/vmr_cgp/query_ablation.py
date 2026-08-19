"""Inference-only counterfactuals for DETR-Query CGP."""

from __future__ import annotations

from typing import Any


def apply_query_cgp_ablation(model: Any, mode: str) -> None:
    query_cgp = getattr(model, "query_cgp", None)
    if query_cgp is None:
        raise ValueError("DETR-Query CGP ablation requires model.query_cgp")
    if mode != "beta_zero":
        raise ValueError(f"Unsupported DETR-Query CGP ablation: {mode}")
    query_cgp.set_beta(0.0)
