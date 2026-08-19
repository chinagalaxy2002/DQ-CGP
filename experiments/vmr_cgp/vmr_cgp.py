"""VMR-specific Compositional Generalization Prompter.

The module is deliberately shape preserving: it consumes projected video and
text tokens and returns an enhanced text sequence with the same ``[B, L, D]``
shape.  Moment-DETR's encoder, decoder, object queries, matcher, and heads are
left unchanged.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class VMRCGPOutput(NamedTuple):
    """Outputs needed by Moment-DETR, auxiliary losses, and diagnostics."""

    enhanced_text: Tensor
    frame_logits: Tensor
    token_frame_logits: Tensor
    basis_weights: Tensor
    prompt_sequence: Tensor
    residual_update: Tensor


class VMRCGP(nn.Module):
    """Adapt text tokens using multi-evidence temporal relations and CGP.

    RCG is performed independently for every text token.  BPS synthesizes a
    short prompt sequence from a shared basis bank.  FRF reads its update only
    from that synthesized prompt, preventing the visual context from bypassing
    the basis prompts through an unconstrained update MLP.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_basis: int = 16,
        prompt_length: int = 4,
        router_hidden_dim: int = 256,
        temperature: float = 1.0,
        alpha_init: float = 0.01,
        alpha_trainable: bool = True,
        gate_floor: float = 0.0,
        logit_scale_init: float = 5.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or num_basis <= 0 or prompt_length <= 0:
            raise ValueError("hidden_dim, num_basis, and prompt_length must be positive")
        if router_hidden_dim <= 0 or temperature <= 0 or logit_scale_init <= 0:
            raise ValueError("router_hidden_dim, temperature, and logit_scale_init must be positive")
        if not 0.0 <= gate_floor < 1.0:
            raise ValueError("gate_floor must lie in [0, 1)")

        self.hidden_dim = int(hidden_dim)
        self.num_basis = int(num_basis)
        self.prompt_length = int(prompt_length)
        self.temperature = float(temperature)
        self.gate_floor = float(gate_floor)

        self.text_norm = nn.LayerNorm(hidden_dim)
        self.video_norm = nn.LayerNorm(hidden_dim)
        self.text_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.video_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.video_value = nn.Linear(hidden_dim, hidden_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(logit_scale_init)))

        router_input_dim = 3 * hidden_dim + 1
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, num_basis),
        )
        self.basis_prompts = nn.Parameter(
            torch.empty(num_basis, prompt_length, hidden_dim)
        )

        # Prompt-only FRF: text/context control selection and gating, while the
        # update values themselves can only come from synthesized prompts.
        self.prompt_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.prompt_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.prompt_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.update_norm = nn.LayerNorm(hidden_dim)
        self.feature_gate = nn.Sequential(
            nn.Linear(3 * hidden_dim + 1, router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, hidden_dim),
        )
        self.token_importance = nn.Linear(hidden_dim, 1)
        self.alpha = nn.Parameter(
            torch.tensor(float(alpha_init)), requires_grad=bool(alpha_trainable)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.basis_prompts)
        # Start with a conservative gate without disabling gradient flow.
        nn.init.zeros_(self.feature_gate[-1].weight)
        nn.init.constant_(self.feature_gate[-1].bias, -2.0)

    @staticmethod
    def _check_inputs(
        video: Tensor,
        video_mask: Tensor,
        text: Tensor,
        text_mask: Tensor,
        hidden_dim: int,
    ) -> None:
        if video.ndim != 3 or text.ndim != 3:
            raise ValueError("video and text must have shape [B, length, hidden_dim]")
        if video.shape[0] != text.shape[0]:
            raise ValueError("video and text batch sizes must match")
        if video.shape[-1] != hidden_dim or text.shape[-1] != hidden_dim:
            raise ValueError(
                f"expected hidden_dim={hidden_dim}, got video={video.shape[-1]} "
                f"and text={text.shape[-1]}"
            )
        if video_mask.shape != video.shape[:2] or text_mask.shape != text.shape[:2]:
            raise ValueError("mask shapes must match the first two feature dimensions")

    def _relation_logits(self, text: Tensor, video: Tensor) -> Tensor:
        text_key = F.normalize(self.text_key(self.text_norm(text)), dim=-1)
        video_key = F.normalize(self.video_key(self.video_norm(video)), dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * torch.einsum("bld,btd->blt", text_key, video_key)

    def forward(
        self,
        video: Tensor,
        video_mask: Tensor,
        text: Tensor,
        text_mask: Tensor,
    ) -> VMRCGPOutput:
        self._check_inputs(video, video_mask, text, text_mask, self.hidden_dim)
        video_valid = video_mask.bool()
        text_valid = text_mask.bool()
        video_weights = video_valid.to(video.dtype)
        text_weights = text_valid.to(text.dtype)

        token_frame_logits = self._relation_logits(text, video)
        valid_relations = text_valid.unsqueeze(-1) & video_valid.unsqueeze(1)
        relation = torch.sigmoid(token_frame_logits) * valid_relations.to(text.dtype)

        # Divide by valid video length rather than relevance mass.  This keeps
        # the amount of matching evidence, so an all-negative/null query tends
        # toward a small context instead of being force-normalized.
        video_values = self.video_value(self.video_norm(video))
        context = torch.einsum("blt,btd->bld", relation, video_values)
        valid_video_count = video_weights.sum(dim=1).clamp_min(1.0)
        context = context / valid_video_count[:, None, None]
        evidence = relation.sum(dim=-1, keepdim=True) / valid_video_count[:, None, None]

        router_input = torch.cat([text, context, text * context, evidence], dim=-1)
        router_logits = self.router(router_input)
        basis_weights = torch.softmax(router_logits / self.temperature, dim=-1)
        prompt_sequence = torch.einsum(
            "bln,npd->blpd", basis_weights, self.basis_prompts
        )

        prompt_query = self.prompt_query(self.text_norm(text)).unsqueeze(-2)
        prompt_keys = self.prompt_key(prompt_sequence)
        prompt_values = self.prompt_value(prompt_sequence)
        prompt_scores = (prompt_query * prompt_keys).sum(dim=-1) / math.sqrt(self.hidden_dim)
        prompt_attention = torch.softmax(prompt_scores, dim=-1)
        prompt_update = torch.einsum("blp,blpd->bld", prompt_attention, prompt_values)
        prompt_update = self.update_norm(prompt_update)

        gate_input = torch.cat([text, context, prompt_update, evidence], dim=-1)
        raw_gate = torch.sigmoid(self.feature_gate(gate_input))
        gate = self.gate_floor + (1.0 - self.gate_floor) * raw_gate
        residual_update = gate * prompt_update * text_weights.unsqueeze(-1)
        enhanced_text = text + self.alpha * residual_update

        # The auxiliary temporal objective is computed from the enhanced text,
        # so it directly supervises the feature that enters Moment-DETR.
        enhanced_token_frame_logits = self._relation_logits(enhanced_text, video)
        token_importance_logits = self.token_importance(enhanced_text).squeeze(-1)
        token_importance_logits = token_importance_logits.masked_fill(
            ~text_valid, torch.finfo(token_importance_logits.dtype).min
        )
        token_importance = torch.softmax(token_importance_logits, dim=1)
        frame_logits = torch.einsum(
            "bl,blt->bt", token_importance, enhanced_token_frame_logits
        )
        frame_logits = frame_logits.masked_fill(~video_valid, -20.0)

        return VMRCGPOutput(
            enhanced_text=enhanced_text,
            frame_logits=frame_logits,
            token_frame_logits=enhanced_token_frame_logits,
            basis_weights=basis_weights,
            prompt_sequence=prompt_sequence,
            residual_update=residual_update,
        )
