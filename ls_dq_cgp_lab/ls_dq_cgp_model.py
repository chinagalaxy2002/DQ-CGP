"""Late-Semantic DQ-CGP Model Wrapper and Loss Integration."""

from __future__ import annotations

from types import MethodType
from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.moment_detr_gmr.utils.span_utils import span_cxw_to_xx
from ls_dq_cgp_lab.cgp_module import LateSemanticCGP, LSDQCGPOutput


class NativeD1AttentionCapture:
    """Capture native D1 cross-attention and video mask from Moment-DETR decoder."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.attention: Optional[Tensor] = None
        self.video_mask: Optional[Tensor] = None
        self.video_width: Optional[int] = None
        decoder = model.transformer.decoder
        self.module = decoder.layers[0].multihead_attn
        self.handle = self.module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        del module, inputs
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise RuntimeError("Native D1 MultiheadAttention did not return weights")
        weights = output[1]
        if weights is None or weights.ndim != 3:
            raise RuntimeError(f"Expected [B,Q,S] attention, got {None if weights is None else weights.shape}")
        self.attention = weights

    def record_input(self, src_vid: Tensor, src_vid_mask: Tensor) -> None:
        self.attention = None
        self.video_mask = src_vid_mask.bool()
        self.video_width = int(src_vid.shape[1])

    def video_attention(self) -> Tensor:
        if self.attention is None or self.video_mask is None or self.video_width is None:
            raise RuntimeError("No native attention captured for current forward pass")
        attention = self.attention[:, :, : self.video_width]
        valid = self.video_mask[:, : self.video_width].unsqueeze(1).to(attention.dtype)
        attention = attention * valid
        return attention / attention.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(attention.dtype).eps
        )

    def remove(self) -> None:
        self.handle.remove()


def _overlap(spans: Tensor, valid_length: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    xx = span_cxw_to_xx(spans).clamp(0.0, 1.0)
    starts = torch.arange(valid_length, dtype=dtype, device=device) / float(valid_length)
    ends = starts + 1.0 / float(valid_length)
    overlap = (starts.unsqueeze(0) < xx[:, 1:]) & (ends.unsqueeze(0) > xx[:, :1])
    empty = ~overlap.any(dim=1)
    if bool(empty.any()):
        centers = 0.5 * (starts + ends)
        nearest = (centers.unsqueeze(0) - xx[:, :1]).abs().argmin(dim=1)
        overlap[empty] = False
        overlap[empty, nearest[empty]] = True
    return overlap


def native_matched_binding_loss(
    attention: Tensor, video_mask: Tensor, targets: Dict, indices: list
) -> Tensor:
    """Production-equivalent Hungarian matched GT-mass loss on D1 attention."""
    terms = []
    for batch_index, (src_indices, target_indices) in enumerate(indices):
        if src_indices.numel() == 0:
            continue
        valid_length = int(video_mask[batch_index].sum().item())
        if valid_length <= 0:
            continue
        src_indices = src_indices.to(attention.device)
        target_indices = target_indices.to(attention.device)
        spans = targets["span_labels"][batch_index]["spans"][target_indices].to(attention.device)
        positive = _overlap(spans, valid_length, attention.dtype, attention.device)
        mass = (
            attention[batch_index, src_indices, :valid_length]
            * positive.to(attention.dtype)
        ).sum(dim=1)
        terms.append(-mass.clamp_min(torch.finfo(attention.dtype).eps).log())
    return torch.cat(terms).mean() if terms else attention.sum() * 0.0


class LSDQCGPModel(nn.Module):
    """Wrapper that equips plain Moment-DETR with Late-Semantic DQ-CGP."""

    def __init__(
        self,
        base_model: nn.Module,
        num_basis: int = 16,
        prompt_length: int = 6,
        router_hidden_dim: int = 256,
        frf_hidden_dim: int = 512,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.capture = NativeD1AttentionCapture(base_model)
        self.cgp = LateSemanticCGP(
            hidden_dim=base_model.transformer.d_model,
            num_basis=num_basis,
            prompt_length=prompt_length,
            router_hidden_dim=router_hidden_dim,
            frf_hidden_dim=frf_hidden_dim,
            temperature=temperature,
        )
        self.static_bypass = False

    def forward(
        self,
        src_txt: Tensor,
        src_txt_mask: Tensor,
        src_vid: Tensor,
        src_vid_mask: Tensor,
        src_aud: Optional[Tensor] = None,
        src_aud_mask: Optional[Tensor] = None,
        src_txt_semantic_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        self.capture.record_input(src_vid, src_vid_mask)

        # 1. Project inputs
        if src_aud is not None:
            src_vid = torch.cat([src_vid, src_aud], dim=2)
        v_proj = self.base_model.input_vid_proj(src_vid)
        t_proj = self.base_model.input_txt_proj(src_txt)

        # 2. Extract static text semantics (E_static)
        semantic_mask = (
            src_txt_mask.bool()
            if src_txt_semantic_mask is None
            else src_txt_semantic_mask.bool() & src_txt_mask.bool()
        )
        semantic_count = semantic_mask.sum(dim=1, keepdim=True).clamp_min(1)
        semantic_weights = semantic_mask.to(t_proj.dtype).unsqueeze(-1)
        static_semantic = (t_proj * semantic_weights).sum(dim=1) / semantic_count.to(t_proj.dtype)

        # 3. Concatenate and pass through Transformer backbone
        src = torch.cat([v_proj, t_proj], dim=1)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()
        pos_vid = self.base_model.position_embed(v_proj, src_vid_mask)
        pos_txt = (
            self.base_model.txt_position_embed(t_proj)
            if self.base_model.use_txt_pos
            else torch.zeros_like(t_proj)
        )
        pos = torch.cat([pos_vid, pos_txt], dim=1)

        # hs: [#layers, bsz, #queries, d], memory: [bsz, L_total, d]
        hs, memory = self.base_model.transformer(src, ~mask, self.base_model.query_embed.weight, pos)

        # 4. Span coordinates from standard span head
        outputs_coord = self.base_model.span_embed(hs)
        if self.base_model.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()

        # 5. Extract local visual context V_q from bound D1 attention
        d1_attention = self.capture.video_attention()            # [B, Q, T_vid]
        vid_mem = memory[:, :v_proj.shape[1]]                     # [B, T_vid, d]
        v_context = torch.bmm(d1_attention, vid_mem)              # [B, Q, d]

        # 6. Apply Late-Semantic CGP to produce pred_logits
        d2_queries = hs[-1]                                       # [B, Q, d]
        cgp_out: LSDQCGPOutput = self.cgp(
            visual_context=v_context,
            static_semantic=static_semantic,
            query_states=d2_queries,
            static_bypass=self.static_bypass,
        )

        out = {
            "pred_logits": cgp_out.pred_logits,
            "pred_spans": outputs_coord[-1],
            "basis_weights": cgp_out.basis_weights,
            "semantic_scores": cgp_out.semantic_scores,
        }

        # 7. Saliency
        out["saliency_scores"] = self.base_model.saliency_proj(vid_mem).squeeze(-1)

        # 8. Aux outputs if enabled
        if self.base_model.aux_loss:
            aux_logits = self.base_model.class_embed(hs[:-1])
            out["aux_outputs"] = [
                {"pred_logits": a, "pred_spans": b}
                for a, b in zip(aux_logits, outputs_coord[:-1])
            ]

        return out


def install_ls_dq_cgp_loss(criterion: nn.Module, model: LSDQCGPModel, coefficient: float = 0.2) -> None:
    """Install Hungarian matched native binding loss on SetCriterion."""
    original_forward = criterion.forward

    def controlled_forward(this, outputs, targets):
        losses = original_forward(outputs, targets)
        final_outputs = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = this.matcher(final_outputs, targets)
        attention = model.capture.video_attention()
        losses["loss_native_bind"] = native_matched_binding_loss(
            attention, model.capture.video_mask, targets, indices
        )
        return losses

    criterion.forward = MethodType(controlled_forward, criterion)
    criterion.weight_dict["loss_native_bind"] = float(coefficient)
    criterion._ls_dq_cgp_original_forward = original_forward
