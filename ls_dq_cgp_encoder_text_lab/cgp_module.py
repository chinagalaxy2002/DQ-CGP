"""Encoder-Text Late-Semantic DQ-CGP.

Pipeline:
    bound visual context V_q + masked-mean encoder text E_enc
        -> RCG -> BPS (mean pool) -> FRF
        -> E_static + semantic delta
        -> cosine matching with the final D2 query state

E_static remains the stable pre-encoder anchor. E_enc is used only as the
contextualized semantic condition for RCG and FRF.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class EncoderTextLSDQCGPOutput(NamedTuple):
    adapted_semantic: Tensor       # [B, Q, D]
    condition_semantic: Tensor     # [B, Q, D]
    basis_weights: Tensor          # [B, Q, K]
    pooled_prompt: Tensor          # [B, Q, D]
    semantic_scores: Tensor        # [B, Q]
    pred_logits: Tensor            # [B, Q, 2]


class EncoderTextLateSemanticCGP(nn.Module):
    """Bind -> contextualized sentence condition -> adapt -> match."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_basis: int = 16,
        prompt_length: int = 6,
        router_hidden_dim: int = 256,
        frf_hidden_dim: int = 512,
        temperature: float = 1.0,
        initial_scale: float = 10.0,
        initial_bias: float = -2.0,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.num_basis = int(num_basis)
        self.prompt_length = int(prompt_length)
        self.temperature = float(temperature)

        self.router = nn.Sequential(
            nn.Linear(2 * hidden_dim, router_hidden_dim),
            nn.ReLU(),
            nn.Linear(router_hidden_dim, num_basis),
        )

        self.basis_prompts = nn.Parameter(
            torch.empty(num_basis, prompt_length, hidden_dim)
        )
        self.basis_norm = nn.LayerNorm(hidden_dim)

        self.frf_v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.frf = nn.Sequential(
            nn.Linear(3 * hidden_dim, frf_hidden_dim),
            nn.ReLU(),
            nn.Linear(frf_hidden_dim, hidden_dim),
        )
        self.frf_norm = nn.LayerNorm(hidden_dim)

        self.visual_proj = nn.Linear(hidden_dim, hidden_dim)
        self.semantic_proj = nn.Linear(hidden_dim, hidden_dim)
        self.logit_scale = nn.Parameter(torch.tensor(float(initial_scale)))
        self.logit_bias = nn.Parameter(torch.tensor(float(initial_bias)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.basis_prompts)
        for module in (self.router, self.frf):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.frf_v_proj.weight)
        nn.init.zeros_(self.frf_v_proj.bias)
        nn.init.xavier_uniform_(self.visual_proj.weight)
        nn.init.zeros_(self.visual_proj.bias)
        nn.init.xavier_uniform_(self.semantic_proj.weight)
        nn.init.zeros_(self.semantic_proj.bias)

    def forward(
        self,
        visual_context: Tensor,
        static_semantic: Tensor,
        encoder_semantic: Tensor,
        query_states: Tensor,
        pre_encoder_condition: bool = False,
    ) -> EncoderTextLSDQCGPOutput:
        bsz, num_queries, dim = visual_context.shape

        if dim != self.hidden_dim:
            raise ValueError(f"Expected hidden_dim={self.hidden_dim}, got {dim}")
        if static_semantic.shape != (bsz, dim):
            raise ValueError("static_semantic must have shape [B,D]")
        if encoder_semantic.shape != (bsz, dim):
            raise ValueError("encoder_semantic must have shape [B,D]")
        if query_states.shape != visual_context.shape:
            raise ValueError("query_states must have shape [B,Q,D]")

        e_static = static_semantic.unsqueeze(1).expand(bsz, num_queries, dim)
        condition = static_semantic if pre_encoder_condition else encoder_semantic
        e_condition = condition.unsqueeze(1).expand(bsz, num_queries, dim)

        # V_q is conditioning evidence only and remains stop-gradient.
        v_detached = visual_context.detach()
        router_input = torch.cat([v_detached, e_condition], dim=-1)
        router_logits = self.router(router_input)
        basis_weights = torch.softmax(router_logits / self.temperature, dim=-1)

        prompt_seq = torch.einsum(
            "bqk,kpd->bqpd", basis_weights, self.basis_prompts
        )
        # Keep the original BPS mean pooling.
        pooled_prompt = self.basis_norm(prompt_seq.mean(dim=2))

        visual_feature = self.frf_v_proj(v_detached)
        frf_input = torch.cat(
            [pooled_prompt, e_condition, visual_feature], dim=-1
        )
        semantic_delta = self.frf(frf_input)
        adapted_semantic = self.frf_norm(e_static + semantic_delta)

        h_metric = F.normalize(self.visual_proj(query_states), p=2, dim=-1)
        e_metric = F.normalize(
            self.semantic_proj(adapted_semantic), p=2, dim=-1
        )
        cos_sim = (h_metric * e_metric).sum(dim=-1)
        semantic_scores = self.logit_scale * cos_sim + self.logit_bias
        pred_logits = torch.stack(
            [semantic_scores, torch.zeros_like(semantic_scores)], dim=-1
        )

        return EncoderTextLSDQCGPOutput(
            adapted_semantic=adapted_semantic,
            condition_semantic=e_condition,
            basis_weights=basis_weights,
            pooled_prompt=pooled_prompt,
            semantic_scores=semantic_scores,
            pred_logits=pred_logits,
        )
