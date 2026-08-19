"""Temporal Compositional Generalization Prompter (T-CGP).

The module consumes the already projected Moment-DETR video/text tokens.  It
does not replace the language token sequence: its adapted query vector is
appended as one extra text token by ``MomentDETR``.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn


class TemporalCGPOutput(NamedTuple):
    """Diagnostic-rich output returned by :class:`TemporalCGP`."""

    adapted_query: Tensor
    prompt_sequence: Tensor
    coarse_logits: Tensor
    coarse_attention: Tensor
    null_attention: Tensor
    basis_weights: Tensor


def _masked_mean(values: Tensor, valid_mask: Tensor) -> Tensor:
    weights = valid_mask.to(dtype=values.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (values * weights).sum(dim=1) / denominator


class TemporalCGP(nn.Module):
    """APT-inspired, null-aware temporal query adapter.

    Args:
        hidden_dim: Shared projected video/text dimension.
        num_basis: Number of learnable prompt bases.
        prompt_length: Tokens in each basis.  The first experiment uses the
            pooled prompt for FRF; the full sequence is returned for later
            sequence-injection studies.
        router_hidden_dim: Hidden size of the basis router and FRF update MLP.
        temperature: Positive softmax temperature for basis routing.
        alpha_init: Initial scalar strength of the gated residual.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_basis: int = 16,
        prompt_length: int = 1,
        router_hidden_dim: int = 256,
        temperature: float = 1.0,
        alpha_init: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or num_basis <= 0 or prompt_length <= 0:
            raise ValueError("hidden_dim, num_basis, and prompt_length must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.hidden_dim = int(hidden_dim)
        self.num_basis = int(num_basis)
        self.prompt_length = int(prompt_length)
        self.temperature = float(temperature)

        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.video_key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.video_value_projection = nn.Linear(hidden_dim, hidden_dim)

        # The null candidate lets a generalized-retrieval query choose no
        # temporal context instead of forcing softmax mass onto an irrelevant
        # frame.  It is conditioned on both the query and global video.
        self.null_scorer = nn.Sequential(
            nn.Linear(2 * hidden_dim, router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, 1),
        )

        self.router = nn.Sequential(
            nn.Linear(2 * hidden_dim, router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, num_basis),
        )
        self.basis_prompts = nn.Parameter(
            torch.empty(num_basis, prompt_length, hidden_dim)
        )
        self.prompt_norm = nn.LayerNorm(hidden_dim)

        fusion_dim = 3 * hidden_dim
        self.update_mlp = nn.Sequential(
            nn.Linear(fusion_dim, router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, hidden_dim),
        )
        self.feature_gate = nn.Linear(fusion_dim, hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.output_norm = nn.LayerNorm(hidden_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.basis_prompts)
        # Start the learned null candidate without a hard positive/negative
        # prior; final retrieval/existence supervision can calibrate it.
        nn.init.zeros_(self.null_scorer[-1].weight)
        nn.init.zeros_(self.null_scorer[-1].bias)

    def forward(
        self,
        video: Tensor,
        video_mask: Tensor,
        text: Tensor,
        text_mask: Tensor,
    ) -> TemporalCGPOutput:
        if video.ndim != 3 or text.ndim != 3:
            raise ValueError("video and text must have shape [B, length, hidden_dim]")
        if video.shape[0] != text.shape[0]:
            raise ValueError("video and text batch sizes must match")
        if video.shape[-1] != self.hidden_dim or text.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"expected hidden_dim={self.hidden_dim}, got video={video.shape[-1]} "
                f"and text={text.shape[-1]}"
            )
        if video_mask.shape != video.shape[:2] or text_mask.shape != text.shape[:2]:
            raise ValueError("mask shapes must match the first two feature dimensions")

        video_valid = video_mask.bool()
        text_valid = text_mask.bool()
        query = _masked_mean(text, text_valid)
        video_global = _masked_mean(video, video_valid)

        query_key = self.query_projection(query)
        video_keys = self.video_key_projection(video)
        coarse_logits = torch.einsum("btd,bd->bt", video_keys, query_key)
        coarse_logits = coarse_logits / math.sqrt(self.hidden_dim)
        coarse_logits = coarse_logits.masked_fill(~video_valid, float("-inf"))

        null_logit = self.null_scorer(
            torch.cat([query, video_global], dim=-1)
        )
        temporal_and_null = torch.cat([coarse_logits, null_logit], dim=1)
        attention = torch.softmax(temporal_and_null, dim=1)
        coarse_attention = attention[:, :-1]
        null_attention = attention[:, -1]

        video_values = self.video_value_projection(video)
        context = torch.bmm(coarse_attention.unsqueeze(1), video_values).squeeze(1)

        router_logits = self.router(torch.cat([query, context], dim=-1))
        basis_weights = torch.softmax(router_logits / self.temperature, dim=-1)
        prompt_sequence = torch.einsum(
            "bn,nld->bld", basis_weights, self.basis_prompts
        )
        pooled_prompt = self.prompt_norm(prompt_sequence.mean(dim=1))

        fusion = torch.cat([pooled_prompt, query, context], dim=-1)
        update = self.update_mlp(fusion)
        gate = torch.sigmoid(self.feature_gate(fusion))
        relevance = (1.0 - null_attention).unsqueeze(-1)
        adapted_query = self.output_norm(
            query + self.alpha * relevance * gate * update
        )

        return TemporalCGPOutput(
            adapted_query=adapted_query,
            prompt_sequence=prompt_sequence,
            coarse_logits=coarse_logits,
            coarse_attention=coarse_attention,
            null_attention=null_attention,
            basis_weights=basis_weights,
        )
