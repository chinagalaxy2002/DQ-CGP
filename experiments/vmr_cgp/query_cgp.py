"""DETR-query conditioned compositional generalization prompting.

This module is designed to run between two Moment-DETR decoder layers.  The
first decoder layer supplies the candidate-instance axis that APT receives
from object proposals: every native DETR query independently reads a temporal
context, routes over a shared prompt basis, and refines its decoder state.

All decoder-facing tensors use PyTorch's sequence-first transformer layout.
Diagnostics use batch-first layouts so that they can be consumed by losses and
analysis code without another transpose.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import torch
from torch import Tensor, nn


class DETRQueryCGPOutput(NamedTuple):
    """Diagnostic tensors from one active DQ-CGP forward pass.

    Shapes:
        adapted_state: ``[num_queries, batch, hidden_dim]``.
        temporal_logits: ``[batch, num_queries, video_length]``.
        temporal_attention: ``[batch, num_queries, video_length]``.
        temporal_context: ``[batch, num_queries, hidden_dim]``.
        basis_weights: ``[batch, num_queries, num_basis]``.
        prompt_sequence: ``[batch, num_queries, prompt_length, hidden_dim]``.
        pooled_prompt: ``[batch, num_queries, hidden_dim]``.
        frf_feature: ``[batch, num_queries, hidden_dim]``.
        residual_update: ``[batch, num_queries, hidden_dim]``.
    """

    adapted_state: Tensor
    temporal_logits: Tensor
    temporal_attention: Tensor
    temporal_context: Tensor
    basis_weights: Tensor
    prompt_sequence: Tensor
    pooled_prompt: Tensor
    frf_feature: Tensor
    residual_update: Tensor


class DETRQueryCGP(nn.Module):
    """APT-style CGP conditioned on native DETR moment candidates.

    The module implements the following path for every native DETR query:

    ``temporal binding -> RCG -> BPS -> APT-style FRF -> fixed-beta residual``.

    Args:
        hidden_dim: Shared encoder/decoder feature dimension.
        num_basis: Number of shared learnable prompt bases.
        prompt_length: Number of tokens in every basis prompt.
        router_hidden_dim: Hidden dimension of the RCG MLP.
        frf_hidden_dim: Hidden dimension of the APT-style FRF MLP.
        temperature: Softmax temperature for basis routing.
        beta: Fixed residual strength.  It is stored as a non-trainable buffer.
            When it is exactly zero, :meth:`forward` returns its input object
            before validation, projections, attention, or any stochastic work.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_basis: int = 16,
        prompt_length: int = 6,
        router_hidden_dim: int = 256,
        frf_hidden_dim: int = 512,
        temperature: float = 1.0,
        beta: float = 0.05,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_basis <= 0:
            raise ValueError("num_basis must be positive")
        if prompt_length <= 0:
            raise ValueError("prompt_length must be positive")
        if router_hidden_dim <= 0 or frf_hidden_dim <= 0:
            raise ValueError("router_hidden_dim and frf_hidden_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if beta < 0:
            raise ValueError("beta must be non-negative")

        self.hidden_dim = int(hidden_dim)
        self.num_basis = int(num_basis)
        self.prompt_length = int(prompt_length)
        self.temperature = float(temperature)

        # A buffer makes the fixed injection strength checkpoint-visible while
        # keeping it outside optimization.  It can still be zeroed for the
        # inference counterfactual specified by the experiment protocol.
        self.register_buffer("beta", torch.tensor(float(beta)))
        self._beta_is_zero = float(beta) == 0.0

        # Per-candidate temporal binding.
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.candidate_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.semantic_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.memory_key_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.memory_value_projection = nn.Linear(hidden_dim, hidden_dim)

        # RCG: concatenate candidate-specific visual context and the shared
        # static query semantic, as the local APT skeleton concatenates an
        # object's visual and semantic features.
        self.router = nn.Sequential(
            nn.Linear(2 * hidden_dim, router_hidden_dim),
            nn.ReLU(),
            nn.Linear(router_hidden_dim, num_basis),
        )

        # BPS: one shared prompt basis bank for all DETR candidates.
        self.basis_prompts = nn.Parameter(
            torch.empty(num_basis, prompt_length, hidden_dim)
        )

        # FRF follows the local APT skeleton: project the visual context,
        # concatenate pooled prompt, static semantic, and projected context,
        # then map the result back to the shared feature dimension.
        self.frf_context_projection = nn.Linear(hidden_dim, hidden_dim)
        self.frf = nn.Sequential(
            nn.Linear(3 * hidden_dim, frf_hidden_dim),
            nn.ReLU(),
            nn.Linear(frf_hidden_dim, hidden_dim),
        )
        self.residual_projection = nn.Linear(hidden_dim, hidden_dim)
        self.residual_norm = nn.LayerNorm(hidden_dim)

        # This is intentionally not a buffer: it is transient, may retain an
        # autograd graph for the current step, and must never enter checkpoints.
        self.last_output: Optional[DETRQueryCGPOutput] = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the shared prompt bases as in the local APT skeleton."""

        nn.init.xavier_uniform_(self.basis_prompts)

    def set_beta(self, beta: float) -> None:
        """Set the fixed residual strength, including the zero-beta ablation."""

        if beta < 0:
            raise ValueError("beta must be non-negative")
        self.beta.fill_(float(beta))
        self._beta_is_zero = float(beta) == 0.0

    def clear_diagnostics(self) -> None:
        """Release diagnostics (and their previous autograd graph) before a pass."""

        self.last_output = None

    def _load_from_state_dict(self, *args, **kwargs) -> None:
        super()._load_from_state_dict(*args, **kwargs)
        self._beta_is_zero = float(self.beta.detach().cpu()) == 0.0

    def _check_inputs(
        self,
        decoder_state: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Tensor,
        query_semantic: Tensor,
        video_length: int,
    ) -> None:
        if decoder_state.ndim != 3:
            raise ValueError("decoder_state must have shape [num_queries, batch, hidden_dim]")
        if memory.ndim != 3:
            raise ValueError("memory must have shape [source_length, batch, hidden_dim]")
        if query_semantic.ndim != 2:
            raise ValueError("query_semantic must have shape [batch, hidden_dim]")

        num_queries, batch_size, decoder_dim = decoder_state.shape
        source_length, memory_batch, memory_dim = memory.shape
        del num_queries
        if batch_size != memory_batch or batch_size != query_semantic.shape[0]:
            raise ValueError("decoder_state, memory, and query_semantic batch sizes must match")
        if (
            decoder_dim != self.hidden_dim
            or memory_dim != self.hidden_dim
            or query_semantic.shape[1] != self.hidden_dim
        ):
            raise ValueError(
                f"expected hidden_dim={self.hidden_dim}, got decoder={decoder_dim}, "
                f"memory={memory_dim}, semantic={query_semantic.shape[1]}"
            )
        if memory_key_padding_mask.shape != (batch_size, source_length):
            raise ValueError(
                "memory_key_padding_mask must have shape [batch, source_length]"
            )
        if not isinstance(video_length, int):
            raise TypeError("video_length must be an int")
        if video_length <= 0 or video_length > source_length:
            raise ValueError("video_length must lie in [1, source_length]")

    @staticmethod
    def _masked_temporal_softmax(logits: Tensor, padding_mask: Tensor) -> Tensor:
        """Softmax over valid frames, returning zeros for an all-padding row."""

        valid = ~padding_mask.bool()
        masked_logits = logits.masked_fill(
            ~valid.unsqueeze(1), torch.finfo(logits.dtype).min
        )
        attention = torch.softmax(masked_logits, dim=-1)
        attention = attention * valid.unsqueeze(1).to(attention.dtype)
        denominator = attention.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(attention.dtype).eps
        )
        return attention / denominator

    def forward(
        self,
        decoder_state: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Tensor,
        query_semantic: Tensor,
        video_length: int,
    ) -> Tensor:
        """Refine sequence-first DETR candidate states.

        ``memory_key_padding_mask`` follows ``nn.MultiheadAttention`` semantics:
        ``True`` marks padding.  Only the first ``video_length`` memory tokens
        participate in temporal binding; the remaining joint-memory tokens are
        text tokens and are represented by ``query_semantic`` in RCG/FRF.
        """

        # This branch must precede validation and every feature computation so
        # beta=0 is a true identity and consumes no stochastic state.
        if self._beta_is_zero:
            self.last_output = None
            return decoder_state

        self._check_inputs(
            decoder_state,
            memory,
            memory_key_padding_mask,
            query_semantic,
            video_length,
        )

        candidate = decoder_state.transpose(0, 1)  # [B, M, D]
        video_memory = memory[:video_length].transpose(0, 1)  # [B, T, D]
        video_padding_mask = memory_key_padding_mask[:, :video_length].bool()

        candidate_key = self.candidate_projection(self.decoder_norm(candidate))
        semantic_key = self.semantic_projection(query_semantic).unsqueeze(1)
        temporal_query = candidate_key + semantic_key
        temporal_key = self.memory_key_projection(self.memory_norm(video_memory))
        temporal_logits = torch.einsum(
            "bmd,btd->bmt", temporal_query, temporal_key
        ) / math.sqrt(self.hidden_dim)
        temporal_attention = self._masked_temporal_softmax(
            temporal_logits, video_padding_mask
        )

        video_value = self.memory_value_projection(video_memory)
        temporal_context = torch.einsum(
            "bmt,btd->bmd", temporal_attention, video_value
        )

        semantic = query_semantic.unsqueeze(1).expand(
            -1, candidate.shape[1], -1
        )
        router_input = torch.cat([temporal_context, semantic], dim=-1)
        router_logits = self.router(router_input)
        basis_weights = torch.softmax(
            router_logits / self.temperature, dim=-1
        )

        prompt_sequence = torch.einsum(
            "bmn,npd->bmpd", basis_weights, self.basis_prompts
        )
        pooled_prompt = prompt_sequence.mean(dim=2)

        projected_context = self.frf_context_projection(temporal_context)
        frf_input = torch.cat(
            [pooled_prompt, semantic, projected_context], dim=-1
        )
        frf_feature = self.frf(frf_input)
        residual_update = self.residual_norm(
            self.residual_projection(frf_feature)
        )

        adapted_candidate = candidate + self.beta.to(candidate.dtype) * residual_update
        adapted_state = adapted_candidate.transpose(0, 1)

        self.last_output = DETRQueryCGPOutput(
            adapted_state=adapted_state,
            temporal_logits=temporal_logits,
            temporal_attention=temporal_attention,
            temporal_context=temporal_context,
            basis_weights=basis_weights,
            prompt_sequence=prompt_sequence,
            pooled_prompt=pooled_prompt,
            frf_feature=frf_feature,
            residual_update=residual_update,
        )
        return adapted_state


# Short alias for configuration and experiment code that uses the paper name.
DQCGP = DETRQueryCGP


__all__ = ["DETRQueryCGP", "DETRQueryCGPOutput", "DQCGP"]
