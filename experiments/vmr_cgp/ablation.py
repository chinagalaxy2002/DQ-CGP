"""Inference-only counterfactuals for a trained VMR-CGP checkpoint."""

from __future__ import annotations

from typing import Any


def apply_vmr_cgp_ablation(model: Any, mode: str) -> None:
    if getattr(model, "vmr_cgp", None) is None:
        raise ValueError("VMR-CGP ablation requires model.vmr_cgp")
    if mode != "alpha_zero":
        raise ValueError(f"Unsupported VMR-CGP ablation: {mode}")
    model.vmr_cgp.alpha.data.zero_()
