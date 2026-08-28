"""Runtime-only native decoder cross-attention capture.

The production ``TransformerDecoderLayer`` asks multi-head attention for only
the output tensor and discards its weights.  Wrapping each decoder layer's
cross-attention module after checkpoint loading lets the analysis observe the
weights without changing model source code or model arithmetic in the normal
forward path.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class CrossAttentionCapture(nn.Module):
    """Forward-compatible wrapper around one ``nn.MultiheadAttention``."""

    def __init__(self, module: nn.MultiheadAttention) -> None:
        super().__init__()
        self.module = module
        self.last_attention: Tensor | None = None

    def clear(self) -> None:
        self.last_attention = None

    def forward(self, *args: Any, **kwargs: Any) -> tuple[Tensor, Tensor | None]:
        # The decoder layer consumes element zero, so returning the regular
        # (output, weights) pair preserves its behavior.  In eval mode this
        # only exposes weights that were already computed by MHA.
        call_kwargs = dict(kwargs)
        call_kwargs["need_weights"] = True
        call_kwargs["average_attn_weights"] = False
        output, attention = self.module(*args, **call_kwargs)
        self.last_attention = attention.detach()
        return output, attention


def install_decoder_attention_capture(model: nn.Module) -> list[CrossAttentionCapture]:
    """Wrap all decoder cross-attention modules and return the wrappers."""

    decoder = getattr(getattr(model, "transformer", None), "decoder", None)
    layers = getattr(decoder, "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("model has no transformer decoder layers")

    wrappers: list[CrossAttentionCapture] = []
    for layer_index, layer in enumerate(layers):
        cross_attention = getattr(layer, "multihead_attn", None)
        if isinstance(cross_attention, CrossAttentionCapture):
            wrapper = cross_attention
        elif isinstance(cross_attention, nn.MultiheadAttention):
            wrapper = CrossAttentionCapture(cross_attention)
            layer.multihead_attn = wrapper
        else:
            raise TypeError(
                f"decoder layer {layer_index} has unsupported cross-attention module: "
                f"{type(cross_attention)!r}"
            )
        wrapper.clear()
        wrappers.append(wrapper)
    return wrappers


def get_decoder_attention(wrappers: list[CrossAttentionCapture]) -> list[Tensor]:
    """Return detached attention tensors in decoder-layer order."""

    result: list[Tensor] = []
    for layer_index, wrapper in enumerate(wrappers):
        if wrapper.last_attention is None:
            raise RuntimeError(
                f"decoder layer {layer_index} did not produce cross-attention weights"
            )
        result.append(wrapper.last_attention)
    return result


def as_batch_head_query_key(attention: Tensor, batch_size: int) -> Tensor:
    """Normalize PyTorch MHA weight layouts to ``[B, H, Q, S]``."""

    if attention.ndim == 4:
        if attention.shape[0] != batch_size:
            raise ValueError(
                f"unexpected attention batch dimension: {tuple(attention.shape)}"
            )
        return attention
    if attention.ndim == 3:
        # This is the fallback layout when a PyTorch version does not support
        # per-head weights.  It remains useful, with a singleton head axis.
        if attention.shape[0] != batch_size:
            raise ValueError(
                f"unexpected attention batch dimension: {tuple(attention.shape)}"
            )
        return attention.unsqueeze(1)
    raise ValueError(f"unsupported attention shape: {tuple(attention.shape)}")

