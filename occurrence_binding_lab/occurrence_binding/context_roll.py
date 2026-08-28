"""Inference-only DQ-CGP context-permutation counterfactual.

The wrapper leaves temporal attention, the context vectors, and their norms
unchanged as a multiset.  It rolls the candidate axis before the router and
FRF, so candidate ``j`` receives the downstream update computed from context
``j-1``.  This isolates candidate-to-context correspondence.
"""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch

from experiments.vmr_cgp.query_cgp import DETRQueryCGPOutput


def install_context_roll(query_cgp: Any) -> None:
    """Install the context-roll wrapper on one loaded DQ-CGP module.

    The function is deliberately idempotent.  It is applied after checkpoint
    loading, so the wrapper itself never becomes part of a checkpoint state
    dict.
    """

    if getattr(query_cgp, "_occurrence_binding_context_roll", False):
        return
    if not hasattr(query_cgp, "last_output"):
        raise ValueError("context_roll requires a DETRQueryCGP module")

    original_forward = query_cgp.forward

    def rolled_forward(module: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        active_state = original_forward(*args, **kwargs)
        diagnostics = module.last_output
        if diagnostics is None:
            # This is the module's exact beta=0 identity path.  It should not
            # be silently turned into a different counterfactual.
            return active_state

        decoder_state = kwargs.get("decoder_state")
        query_semantic = kwargs.get("query_semantic")
        if decoder_state is None and args:
            decoder_state = args[0]
        if query_semantic is None and len(args) >= 4:
            query_semantic = args[3]
        if decoder_state is None or query_semantic is None:
            raise ValueError(
                "context_roll requires decoder_state and query_semantic inputs"
            )

        candidate = decoder_state.transpose(0, 1)
        effective_context = diagnostics.temporal_context.roll(shifts=1, dims=1)
        semantic = query_semantic.unsqueeze(1).expand(
            -1, candidate.shape[1], -1
        )
        router_input = torch.cat([effective_context, semantic], dim=-1)
        router_logits = module.router(router_input)
        basis_weights = torch.softmax(
            router_logits / module.temperature, dim=-1
        )
        prompt_sequence = torch.einsum(
            "bmn,npd->bmpd", basis_weights, module.basis_prompts
        )
        pooled_prompt = prompt_sequence.mean(dim=2)
        projected_context = module.frf_context_projection(effective_context)
        frf_input = torch.cat(
            [pooled_prompt, semantic, projected_context], dim=-1
        )
        frf_feature = module.frf(frf_input)
        residual_update = module.residual_norm(
            module.residual_projection(frf_feature)
        )
        adapted_candidate = candidate + module.beta.to(candidate.dtype) * residual_update
        adapted_state = adapted_candidate.transpose(0, 1)

        module.last_output = DETRQueryCGPOutput(
            adapted_state=adapted_state,
            temporal_logits=diagnostics.temporal_logits,
            temporal_attention=diagnostics.temporal_attention,
            temporal_context=effective_context,
            basis_weights=basis_weights,
            prompt_sequence=prompt_sequence,
            pooled_prompt=pooled_prompt,
            frf_feature=frf_feature,
            residual_update=residual_update,
        )
        return adapted_state

    query_cgp.forward = MethodType(rolled_forward, query_cgp)
    query_cgp._occurrence_binding_context_roll = True


def apply_context_roll(model: Any) -> None:
    """Install context-roll on ``model.query_cgp``."""

    query_cgp = getattr(model, "query_cgp", None)
    if query_cgp is None:
        raise ValueError("context_roll requires model.query_cgp")
    install_context_roll(query_cgp)

