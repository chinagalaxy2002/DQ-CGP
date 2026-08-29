"""Runtime-only native D1 cross-attention binding supervision."""

from __future__ import annotations

from types import MethodType

import torch

from models.moment_detr_gmr.utils.span_utils import span_cxw_to_xx


class NativeD1AttentionCapture:
    """Capture D1 native cross-attention and the corresponding video mask."""

    def __init__(self, model):
        self.model = model
        self.attention = None
        self.video_mask = None
        self.video_width = None
        decoder = model.transformer.decoder
        self.module = decoder.layers[0].multihead_attn
        self.handle = self.module.register_forward_hook(self._hook)
        self.original_model_forward = model.forward

        def wrapped_forward(this, *args, **kwargs):
            self.attention = None
            mask = kwargs.get("src_vid_mask")
            video = kwargs.get("src_vid")
            if mask is None or video is None:
                raise RuntimeError("Native binding requires src_vid and src_vid_mask")
            self.video_mask = mask.bool()
            self.video_width = int(video.shape[1])
            return self.original_model_forward(*args, **kwargs)

        model.forward = MethodType(wrapped_forward, model)

    def _hook(self, module, inputs, output):
        del module, inputs
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise RuntimeError("Native D1 MultiheadAttention did not return weights")
        weights = output[1]
        if weights is None or weights.ndim != 3:
            raise RuntimeError(f"Expected [B,Q,S] attention, got {None if weights is None else weights.shape}")
        self.attention = weights

    def video_attention(self):
        if self.attention is None or self.video_mask is None or self.video_width is None:
            raise RuntimeError("No native attention captured for the current forward")
        attention = self.attention[:, :, : self.video_width]
        valid = self.video_mask[:, : self.video_width].unsqueeze(1).to(attention.dtype)
        attention = attention * valid
        return attention / attention.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(attention.dtype).eps
        )

    def remove(self):
        self.handle.remove()
        self.model.forward = self.original_model_forward


def _overlap(spans, valid_length, dtype, device):
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


def native_matched_binding_loss(attention, video_mask, targets, indices):
    """Production-equivalent matched GT-mass loss on native D1 attention."""
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


def install_native_binding_loss(criterion, capture, coefficient=0.2):
    """Append native binding loss while preserving production criterion logic."""
    original_forward = criterion.forward

    def controlled_forward(this, outputs, targets):
        losses = original_forward(outputs, targets)
        final_outputs = {key: value for key, value in outputs.items() if key != "aux_outputs"}
        indices = this.matcher(final_outputs, targets)
        attention = capture.video_attention()
        losses["loss_native_bind"] = native_matched_binding_loss(
            attention, capture.video_mask, targets, indices
        )
        return losses

    criterion.forward = MethodType(controlled_forward, criterion)
    criterion.weight_dict["loss_native_bind"] = float(coefficient)
    criterion._native_binding_original_forward = original_forward
