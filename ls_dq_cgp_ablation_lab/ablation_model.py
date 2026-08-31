"""Shape- and checkpoint-compatible component ablations for LS-DQ-CGP.

No production module is modified.  Every variant retains the same parameters
and output interface, so a Full checkpoint can be evaluated inference-only and
each variant can also be trained from scratch under the same configuration.
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from ls_dq_cgp_lab.cgp_module import LateSemanticCGP, LSDQCGPOutput
from ls_dq_cgp_lab.ls_dq_cgp_model import LSDQCGPModel


ABLATION_VARIANTS = (
    "full",
    "rcg_uniform",
    "bps_query_mean",
    "bps_zero",
    "frf_remove",
)


class AblationLateSemanticCGP(LateSemanticCGP):
    """LateSemanticCGP with one explicitly controlled intervention."""

    def __init__(self, *args, variant: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if variant not in ABLATION_VARIANTS:
            raise ValueError(
                f"Unknown variant {variant!r}; choose from {ABLATION_VARIANTS}"
            )
        self.variant = variant

    def forward(
        self,
        visual_context: Tensor,
        static_semantic: Tensor,
        query_states: Tensor,
        static_bypass: bool = False,
    ) -> LSDQCGPOutput:
        if static_bypass:
            raise ValueError(
                "Component ablations cannot be combined with legacy static_bypass"
            )

        batch_size, num_queries, dim = visual_context.shape
        e_static = static_semantic.unsqueeze(1).expand(
            batch_size, num_queries, dim
        )
        v_detached = visual_context.detach()

        gate_logits = self.router(torch.cat([v_detached, e_static], dim=-1))
        routed_weights = F.softmax(
            gate_logits / self.temperature, dim=-1
        )
        if self.variant == "rcg_uniform":
            basis_weights = torch.full_like(
                routed_weights, 1.0 / float(self.num_basis)
            )
        else:
            basis_weights = routed_weights

        bases = self.basis_prompts.unsqueeze(0).unsqueeze(0)
        weights = basis_weights.unsqueeze(-1).unsqueeze(-1)
        prompt_sequence = (weights * bases).sum(dim=2)
        pooled_prompt = self.basis_norm(prompt_sequence.mean(dim=2))

        if self.variant == "bps_query_mean":
            # Mean over DETR candidates, not over bases.  Averaging bases would
            # be exactly the same operation as rcg_uniform and is therefore not
            # a distinct scientific control.
            pooled_prompt = pooled_prompt.mean(dim=1, keepdim=True).expand_as(
                pooled_prompt
            )
        elif self.variant == "bps_zero":
            pooled_prompt = torch.zeros_like(pooled_prompt)

        if self.variant == "frf_remove":
            # Remove the learned fusion MLP while retaining the routed BPS
            # feature as a dimension-compatible residual.  Visual context can
            # still affect predictions through RCG -> BPS.
            delta_semantic = pooled_prompt
        else:
            v_projected = self.frf_v_proj(v_detached)
            frf_input = torch.cat(
                [pooled_prompt, e_static, v_projected], dim=-1
            )
            delta_semantic = self.frf(frf_input)

        e_adapt = self.frf_norm(e_static + delta_semantic)
        h_metric = F.normalize(self.visual_proj(query_states), p=2, dim=-1)
        e_metric = F.normalize(self.semantic_proj(e_adapt), p=2, dim=-1)
        cosine = (h_metric * e_metric).sum(dim=-1)
        semantic_scores = self.logit_scale * cosine + self.logit_bias
        pred_logits = torch.stack(
            [semantic_scores, torch.zeros_like(semantic_scores)], dim=-1
        )

        return LSDQCGPOutput(
            adapted_semantic=e_adapt,
            basis_weights=basis_weights,
            pooled_prompt=pooled_prompt,
            semantic_scores=semantic_scores,
            pred_logits=pred_logits,
        )


class AblationLSDQCGPModel(LSDQCGPModel):
    """Drop-in LS-DQ-CGP wrapper for a selected component ablation."""

    def __init__(self, *args, variant: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        old_cgp = self.cgp
        self.cgp = AblationLateSemanticCGP(
            hidden_dim=old_cgp.hidden_dim,
            num_basis=old_cgp.num_basis,
            prompt_length=old_cgp.prompt_length,
            router_hidden_dim=old_cgp.router[0].out_features,
            frf_hidden_dim=old_cgp.frf[0].out_features,
            temperature=old_cgp.temperature,
            variant=variant,
        )
        self.ablation_variant = variant
        self.static_bypass = False

