"""Inference-only ablations for checking whether T-CGP refinement is useful."""

from __future__ import annotations

from typing import Any


def install_tcgp_ablation(model: Any, mode: str) -> Any:
    """Install an inference hook and return its removable handle.

    ``normalized_query`` keeps the extra-token interface unchanged but removes
    the video-conditioned FRF update.  The appended token becomes LN(Pool(Q)),
    using the learned output LayerNorm from the same checkpoint.
    """
    if mode != "normalized_query":
        raise ValueError(f"Unsupported T-CGP ablation: {mode}")
    if getattr(model, "tcgp", None) is None:
        raise ValueError("T-CGP ablation requires a model with model.tcgp")

    def replace_adapted_query(module, inputs, output):
        text = inputs[2]
        text_mask = inputs[3].to(dtype=text.dtype).unsqueeze(-1)
        pooled_query = (text * text_mask).sum(dim=1) / text_mask.sum(
            dim=1
        ).clamp_min(1.0)
        normalized_query = module.output_norm(pooled_query)
        return output._replace(adapted_query=normalized_query)

    return model.tcgp.register_forward_hook(replace_adapted_query)
