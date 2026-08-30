"""Late-Semantic DETR-Query Compositional Generalization Prompter (LS-DQ-CGP).

Implements late semantic modulation for GMR candidate ranking:
1. RCG: Evaluates query-specific visual context V_q against static semantic E_static.
2. BPS: Synthesizes dynamic prompt from shared learnable basis prompts.
3. FRF: Generates adapted query-specific semantic feature E_adapt^q.
4. Semantic Matcher: Scores cosine similarity between D2 query state h_q and E_adapt^q.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LSDQCGPOutput(NamedTuple):
    """Output container for LS-DQ-CGP forward pass."""
    adapted_semantic: Tensor           # [B, Q, D]
    basis_weights: Tensor              # [B, Q, num_basis]
    pooled_prompt: Tensor              # [B, Q, D]
    semantic_scores: Tensor            # [B, Q]
    pred_logits: Tensor                # [B, Q, 2]


class LateSemanticCGP(nn.Module):
    """Late-Semantic Compositional Generalization Prompter.
    
    Transforms global sentence semantics into query-specific adapted semantics
    conditioned on the detached local visual context V_q.
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
        self.hidden_dim = hidden_dim
        self.num_basis = num_basis
        self.prompt_length = prompt_length
        self.temperature = float(temperature)

        # RCG: Condition on [V_q (detached); E_static]
        self.router = nn.Sequential(
            nn.Linear(2 * hidden_dim, router_hidden_dim),
            nn.ReLU(),
            nn.Linear(router_hidden_dim, num_basis),
        )

        # BPS: Shared learnable basis prompts
        self.basis_prompts = nn.Parameter(
            torch.empty(num_basis, prompt_length, hidden_dim)
        )
        self.basis_norm = nn.LayerNorm(hidden_dim)

        # FRF: Feature Refinement & Fusion
        self.frf_v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.frf = nn.Sequential(
            nn.Linear(3 * hidden_dim, frf_hidden_dim),
            nn.ReLU(),
            nn.Linear(frf_hidden_dim, hidden_dim),
        )
        self.frf_norm = nn.LayerNorm(hidden_dim)

        # Semantic Matcher: Project visual query h_q and adapted text E_adapt to metric space
        self.visual_proj = nn.Linear(hidden_dim, hidden_dim)
        self.semantic_proj = nn.Linear(hidden_dim, hidden_dim)

        # Learnable logit scale and bias for cosine matching
        self.logit_scale = nn.Parameter(torch.tensor(float(initial_scale)))
        self.logit_bias = nn.Parameter(torch.tensor(float(initial_bias)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.basis_prompts)
        for m in (self.router, self.frf):
            for layer in m:
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
        visual_context: Tensor,       # V_q: [B, Q, D] (detached)
        static_semantic: Tensor,      # E_static: [B, D]
        query_states: Tensor,         # h_q: [B, Q, D] from D2
        static_bypass: bool = False,  # If True, counterfactual bypass with E_static
    ) -> LSDQCGPOutput:
        bsz, num_queries, dim = visual_context.shape
        # Expand static semantic across all queries: [B, Q, D]
        e_static_expanded = static_semantic.unsqueeze(1).expand(bsz, num_queries, dim)

        # 1. RCG: Relational Context Gating
        v_detached = visual_context.detach()
        rcg_input = torch.cat([v_detached, e_static_expanded], dim=-1) # [B, Q, 2D]
        gate_logits = self.router(rcg_input)                            # [B, Q, num_basis]
        basis_weights = F.softmax(gate_logits / self.temperature, dim=-1) # [B, Q, num_basis]

        # 2. BPS: Basis Prompt Synthesis
        bases = self.basis_prompts.unsqueeze(0).unsqueeze(0)
        weights = basis_weights.unsqueeze(-1).unsqueeze(-1)             # [B, Q, num_basis, 1, 1]
        prompt_seq = (weights * bases).sum(dim=2)                       # [B, Q, prompt_length, dim]
        pooled_prompt = self.basis_norm(prompt_seq.mean(dim=2))         # [B, Q, dim]

        # 3. FRF: Feature Refinement & Fusion
        v_proj = self.frf_v_proj(v_detached)                           # [B, Q, dim]
        frf_input = torch.cat([pooled_prompt, e_static_expanded, v_proj], dim=-1) # [B, Q, 3D]
        e_adapt = self.frf_norm(e_static_expanded + self.frf(frf_input))          # [B, Q, dim]

        # 4. Semantic Matching Score
        h_metric = F.normalize(self.visual_proj(query_states), p=2, dim=-1) # [B, Q, D]

        if static_bypass:
            e_metric = F.normalize(self.semantic_proj(e_static_expanded), p=2, dim=-1)
        else:
            e_metric = F.normalize(self.semantic_proj(e_adapt), p=2, dim=-1)

        cos_sim = (h_metric * e_metric).sum(dim=-1)                     # [B, Q]
        semantic_scores = self.logit_scale * cos_sim + self.logit_bias   # [B, Q]

        zeros = torch.zeros_like(semantic_scores)
        pred_logits = torch.stack([semantic_scores, zeros], dim=-1)     # [B, Q, 2]

        return LSDQCGPOutput(
            adapted_semantic=e_adapt,
            basis_weights=basis_weights,
            pooled_prompt=pooled_prompt,
            semantic_scores=semantic_scores,
            pred_logits=pred_logits,
        )
