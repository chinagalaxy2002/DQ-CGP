"""Strict inference-only ``Delta E_q = 0`` ablation.

This file intentionally does not change the production LS-DQ-CGP code.  It
reuses the trained ``frf_norm``, ``semantic_proj``, visual projection, cosine
matcher, logit scale, and logit bias, while replacing only the learned semantic
residual with zero:

    E_adapt^q = frf_norm(E_static + 0).
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from ls_dq_cgp_lab.cgp_module import LateSemanticCGP, LSDQCGPOutput
from ls_dq_cgp_lab.ls_dq_cgp_model import LSDQCGPModel


class StrictDeltaZeroCGP(LateSemanticCGP):
    """Evaluate the exact zero-residual semantic counterfactual."""

    def forward(
        self,
        visual_context: Tensor,
        static_semantic: Tensor,
        query_states: Tensor,
        static_bypass: bool = False,
    ) -> LSDQCGPOutput:
        if static_bypass:
            raise ValueError(
                "StrictDeltaZeroCGP and legacy static_bypass are mutually exclusive"
            )

        # Preserve the normal diagnostic tensors.  Predictions below do not
        # depend on the residual computed by this call.
        active = super().forward(
            visual_context=visual_context,
            static_semantic=static_semantic,
            query_states=query_states,
            static_bypass=False,
        )

        _, num_queries, _ = visual_context.shape
        e_static_expanded = static_semantic.unsqueeze(1).expand(
            -1, num_queries, -1
        )

        # The only intervention: Delta E_q is identically zero.  The learned
        # LayerNorm affine parameters remain active, unlike legacy bypass.
        e_adapt = self.frf_norm(e_static_expanded)

        # Keep the trained prediction path exactly as in active inference.
        h_metric = F.normalize(self.visual_proj(query_states), p=2, dim=-1)
        e_metric = F.normalize(self.semantic_proj(e_adapt), p=2, dim=-1)
        cos_sim = (h_metric * e_metric).sum(dim=-1)
        semantic_scores = self.logit_scale * cos_sim + self.logit_bias
        pred_logits = torch.stack(
            [semantic_scores, torch.zeros_like(semantic_scores)], dim=-1
        )

        return LSDQCGPOutput(
            adapted_semantic=e_adapt,
            basis_weights=active.basis_weights,
            pooled_prompt=active.pooled_prompt,
            semantic_scores=semantic_scores,
            pred_logits=pred_logits,
        )


class StrictDeltaZeroLSDQCGPModel(LSDQCGPModel):
    """Drop-in LS-DQ-CGP wrapper using :class:`StrictDeltaZeroCGP`."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        old_cgp = self.cgp
        self.cgp = StrictDeltaZeroCGP(
            hidden_dim=old_cgp.hidden_dim,
            num_basis=old_cgp.num_basis,
            prompt_length=old_cgp.prompt_length,
            router_hidden_dim=old_cgp.router[0].out_features,
            frf_hidden_dim=old_cgp.frf[0].out_features,
            temperature=old_cgp.temperature,
        )
        self.static_bypass = False

