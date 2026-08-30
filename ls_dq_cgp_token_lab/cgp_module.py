"""Token-Selective Late-Semantic DQ-CGP.

Pipeline:
    bound visual context V_q
        -> occurrence-conditioned text token selection
        -> local semantic E_local^q
        -> RCG -> BPS -> FRF
        -> adapted semantic E_adapt^q
        -> cosine matching with final D2 query state
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TokenLSDQCGPOutput(NamedTuple):
    adapted_semantic: Tensor       # [B, Q, D]
    local_semantic: Tensor         # [B, Q, D]
    token_attention: Tensor        # [B, Q, L]
    basis_weights: Tensor          # [B, Q, K]
    pooled_prompt: Tensor          # [B, Q, D]
    semantic_scores: Tensor        # [B, Q]
    pred_logits: Tensor            # [B, Q, 2]


class TokenSelectiveLateSemanticCGP(nn.Module):
    """Bind -> Select -> Adapt -> Match.

    V_q is conditioning evidence only and is stop-gradient. Text tokens remain
    trainable through the normal text projection path.
    """

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

        self.selector_visual_norm = nn.LayerNorm(hidden_dim)
        self.selector_text_norm = nn.LayerNorm(hidden_dim)
        self.selector_visual_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.selector_text_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.local_semantic_norm = nn.LayerNorm(hidden_dim)

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
        nn.init.xavier_uniform_(self.selector_visual_proj.weight)
        nn.init.xavier_uniform_(self.selector_text_proj.weight)

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

    @staticmethod
    def _masked_softmax(logits: Tensor, valid_mask: Tensor) -> Tensor:
        """Apply softmax over valid text tokens only.

        Args:
            logits: [B, Q, L].
            valid_mask: [B, L], where True marks a selectable token.
        """
        valid_mask = valid_mask.bool()
        masked_logits = logits.masked_fill(
            ~valid_mask.unsqueeze(1), torch.finfo(logits.dtype).min
        )
        attention = torch.softmax(masked_logits, dim=-1)
        attention = attention * valid_mask.unsqueeze(1).to(attention.dtype)
        denominator = attention.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(attention.dtype).eps
        )
        return attention / denominator

    def _select_local_semantic(
        self,
        visual_context: Tensor,
        text_tokens: Tensor,
        text_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Use each bound V_q to read query-specific text semantics."""
        v_detached = visual_context.detach()
        visual_query = self.selector_visual_proj(
            self.selector_visual_norm(v_detached)
        )
        text_keys = self.selector_text_proj(self.selector_text_norm(text_tokens))
        token_logits = torch.einsum(
            "bqd,bld->bql", visual_query, text_keys
        ) / math.sqrt(self.hidden_dim)
        token_attention = self._masked_softmax(token_logits, text_mask)

        # Values deliberately use the original projected text tokens.
        local_semantic = torch.einsum(
            "bql,bld->bqd", token_attention, text_tokens
        )
        local_semantic = self.local_semantic_norm(local_semantic)
        return local_semantic, token_attention

    def forward(
        self,
        visual_context: Tensor,
        static_semantic: Tensor,
        text_tokens: Tensor,
        text_mask: Tensor,
        query_states: Tensor,
        static_bypass: bool = False,
        token_static_bypass: bool = False,
    ) -> TokenLSDQCGPOutput:
        bsz, num_queries, dim = visual_context.shape

        if dim != self.hidden_dim:
            raise ValueError(f"Expected hidden_dim={self.hidden_dim}, got {dim}")
        if text_tokens.ndim != 3:
            raise ValueError("text_tokens must have shape [B,L,D]")
        if text_tokens.shape[0] != bsz or text_tokens.shape[2] != dim:
            raise ValueError("text_tokens batch/feature dimensions must match visual_context")
        if static_semantic.shape != (bsz, dim):
            raise ValueError("static_semantic must have shape [B,D]")
        if query_states.shape != visual_context.shape:
            raise ValueError("query_states must have shape [B,Q,D]")
        if text_mask.shape != text_tokens.shape[:2]:
            raise ValueError("text_mask must have shape [B,L]")
        if bool((text_mask.sum(dim=1) == 0).any()):
            raise ValueError("Every sample must contain at least one valid text token")

        e_static = static_semantic.unsqueeze(1).expand(bsz, num_queries, dim)

        local_semantic, token_attention = self._select_local_semantic(
            visual_context, text_tokens, text_mask
        )
        if token_static_bypass:
            local_semantic = e_static

        v_detached = visual_context.detach()
        router_input = torch.cat([v_detached, local_semantic], dim=-1)
        router_logits = self.router(router_input)
        basis_weights = torch.softmax(router_logits / self.temperature, dim=-1)

        prompt_seq = torch.einsum(
            "bqk,kpd->bqpd", basis_weights, self.basis_prompts
        )
        pooled_prompt = self.basis_norm(prompt_seq.mean(dim=2))

        visual_feature = self.frf_v_proj(v_detached)
        frf_input = torch.cat(
            [pooled_prompt, local_semantic, visual_feature], dim=-1
        )
        semantic_delta = self.frf(frf_input)
        adapted_semantic = self.frf_norm(e_static + semantic_delta)

        h_metric = F.normalize(self.visual_proj(query_states), p=2, dim=-1)
        semantic_for_match = e_static if static_bypass else adapted_semantic
        e_metric = F.normalize(
            self.semantic_proj(semantic_for_match), p=2, dim=-1
        )
        cos_sim = (h_metric * e_metric).sum(dim=-1)
        semantic_scores = self.logit_scale * cos_sim + self.logit_bias
        pred_logits = torch.stack(
            [semantic_scores, torch.zeros_like(semantic_scores)], dim=-1
        )

        return TokenLSDQCGPOutput(
            adapted_semantic=adapted_semantic,
            local_semantic=local_semantic,
            token_attention=token_attention,
            basis_weights=basis_weights,
            pooled_prompt=pooled_prompt,
            semantic_scores=semantic_scores,
            pred_logits=pred_logits,
        )
