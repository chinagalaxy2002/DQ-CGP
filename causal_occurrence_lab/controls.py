"""Runtime-only controls for the causal ablations.

The release model is intentionally untouched.  The functions here patch only
the model/criterion instances created in the current Python process.
"""

from __future__ import annotations

import math
from types import MethodType
from typing import Any, Mapping

import torch
from torch import Tensor

from experiments.vmr_cgp.query_cgp import DETRQueryCGPOutput
from models.moment_detr_gmr.utils.span_utils import span_cxw_to_xx


def install_query_cgp_controls(model: Any, *, inject_residual: bool) -> None:
    """Make DQ-CGP compute diagnostics even when injection is disabled."""

    query_cgp = getattr(model, "query_cgp", None)
    if query_cgp is None:
        raise ValueError("query-cgp controls require model.query_cgp")
    query_cgp._causal_inject_residual = bool(inject_residual)
    if getattr(query_cgp, "_causal_forward_installed", False):
        return

    def forward(module: Any, decoder_state: Tensor, memory: Tensor,
                memory_key_padding_mask: Tensor, query_semantic: Tensor,
                video_length: int) -> Tensor:
        # This is the release forward path with only the beta-zero early return
        # removed and the final residual injection made configurable.
        module._check_inputs(
            decoder_state, memory, memory_key_padding_mask,
            query_semantic, video_length,
        )
        candidate = decoder_state.transpose(0, 1)
        video_memory = memory[:video_length].transpose(0, 1)
        video_padding_mask = memory_key_padding_mask[:, :video_length].bool()

        candidate_key = module.candidate_projection(module.decoder_norm(candidate))
        semantic_key = module.semantic_projection(query_semantic).unsqueeze(1)
        temporal_query = candidate_key + semantic_key
        temporal_key = module.memory_key_projection(module.memory_norm(video_memory))
        temporal_logits = torch.einsum(
            "bmd,btd->bmt", temporal_query, temporal_key
        ) / math.sqrt(module.hidden_dim)
        temporal_attention = module._masked_temporal_softmax(
            temporal_logits, video_padding_mask
        )
        video_value = module.memory_value_projection(video_memory)
        temporal_context = torch.einsum(
            "bmt,btd->bmd", temporal_attention, video_value
        )

        semantic = query_semantic.unsqueeze(1).expand(
            -1, candidate.shape[1], -1
        )
        router_input = torch.cat([temporal_context, semantic], dim=-1)
        router_logits = module.router(router_input)
        basis_weights = torch.softmax(
            router_logits / module.temperature, dim=-1
        )
        prompt_sequence = torch.einsum(
            "bmn,npd->bmpd", basis_weights, module.basis_prompts
        )
        pooled_prompt = prompt_sequence.mean(dim=2)
        projected_context = module.frf_context_projection(temporal_context)
        frf_input = torch.cat(
            [pooled_prompt, semantic, projected_context], dim=-1
        )
        frf_feature = module.frf(frf_input)
        residual_update = module.residual_norm(
            module.residual_projection(frf_feature)
        )
        if module._causal_inject_residual:
            adapted_candidate = candidate + module.beta.to(candidate.dtype) * residual_update
        else:
            adapted_candidate = candidate
        adapted_state = adapted_candidate.transpose(0, 1)

        module.last_output = DETRQueryCGPOutput(
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

    query_cgp.forward = MethodType(forward, query_cgp)
    query_cgp._causal_forward_installed = True


def _overlap_mask(
    spans: Tensor,
    valid_length: int,
    span_loss_type: str,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if spans.numel() == 0:
        return torch.zeros((0, valid_length), dtype=torch.bool, device=device)
    if span_loss_type == "l1":
        target_xx = span_cxw_to_xx(spans).clamp(0.0, 1.0)
        starts = torch.arange(valid_length, device=device, dtype=dtype) / float(valid_length)
        ends = starts + 1.0 / float(valid_length)
        overlap = (
            (starts.unsqueeze(0) < target_xx[:, 1:])
            & (ends.unsqueeze(0) > target_xx[:, :1])
        )
        empty = ~overlap.any(dim=1)
        if bool(empty.any()):
            centers = 0.5 * (starts + ends)
            nearest = (centers.unsqueeze(0) - target_xx[:, :1]).abs().argmin(dim=1)
            overlap[empty] = False
            overlap[empty, nearest[empty]] = True
        return overlap
    indices = torch.arange(valid_length, device=device).unsqueeze(0)
    return (indices >= spans[:, :1]) & (indices <= spans[:, 1:])


def _target_indices_for_mode(
    target_indices: Tensor,
    num_targets: int,
    mode: str,
) -> Tensor:
    if mode == "matched":
        return target_indices
    if mode == "rolled":
        return target_indices.roll(1) if num_targets > 1 else target_indices
    if mode == "union":
        # A sentinel is used by the caller; all target windows are selected.
        return torch.arange(num_targets, device=target_indices.device)
    raise ValueError(f"unsupported binding target: {mode}")


def _binding_loss(
    outputs: Mapping[str, Tensor],
    targets: Mapping[str, Any],
    indices: list[tuple[Tensor, Tensor]],
    *,
    attention_key: str,
    video_mask_key: str,
    target_mode: str,
    span_loss_type: str,
) -> Tensor:
    attention = outputs[attention_key]
    video_mask = outputs[video_mask_key].bool()
    terms = []
    eps = torch.finfo(attention.dtype).eps
    for batch_index, (src_indices, target_indices) in enumerate(indices):
        if src_indices.numel() == 0:
            continue
        valid_length = int(video_mask[batch_index].sum().item())
        if valid_length <= 0:
            continue
        src_indices = src_indices.to(attention.device)
        target_indices = target_indices.to(attention.device)
        matched_attention = attention[batch_index, src_indices, :valid_length]
        span_item = targets["span_labels"][batch_index]["spans"]
        span_item = span_item.to(attention.device)
        if target_mode == "union":
            selected_spans = span_item
        else:
            selected_indices = _target_indices_for_mode(
                target_indices, len(span_item), target_mode
            )
            selected_spans = span_item[selected_indices]
        overlap = _overlap_mask(
            selected_spans,
            valid_length,
            span_loss_type,
            attention.dtype,
            attention.device,
        )
        if overlap.numel() == 0:
            continue
        if target_mode == "union":
            positive = overlap.any(dim=0).unsqueeze(0).expand(matched_attention.shape[0], -1)
        else:
            positive = overlap
        target_mass = (matched_attention * positive.to(attention.dtype)).sum(dim=1)
        terms.append(-target_mass.clamp_min(eps).log())
    if terms:
        return torch.cat(terms).mean()
    return attention.sum() * 0.0


def _matched_route_loss(
    routes: Tensor,
    indices: list[tuple[Tensor, Tensor]],
) -> Tensor:
    """Match the production ``SetCriterion.loss_query_cgp`` route objective.

    The marginal distribution must be computed once over all matched queries
    in the batch.  In particular, the objective is

    ``H(route | matched query) - H(mean(route over matched queries))``.

    Keeping this as a small standalone helper makes the causal patch auditable
    and lets the tests compare it directly with the unmodified criterion.
    """

    matched_routes = []
    for batch_index, (src_indices, _) in enumerate(indices):
        if src_indices.numel():
            matched_routes.append(routes[batch_index, src_indices.to(routes.device)])
    if not matched_routes:
        return routes.sum() * 0.0

    selected = torch.cat(matched_routes, dim=0)
    eps = torch.finfo(selected.dtype).eps
    conditional_entropy = -(
        selected * selected.clamp_min(eps).log()
    ).sum(dim=-1).mean()
    marginal = selected.mean(dim=0)
    marginal_entropy = -(
        marginal * marginal.clamp_min(eps).log()
    ).sum()
    return conditional_entropy - marginal_entropy


def install_criterion_controls(
    criterion: Any,
    *,
    binding_target: str = "matched",
    native_binding: bool = False,
) -> None:
    """Install matched/union/rolled binding and optional native binding loss."""

    if binding_target not in {"matched", "union", "rolled"}:
        raise ValueError(f"unsupported binding target: {binding_target}")
    criterion._causal_binding_target = binding_target
    criterion._causal_native_binding = bool(native_binding)
    if getattr(criterion, "_causal_loss_installed", False):
        return

    original_get_loss = criterion.get_loss

    def get_loss(self: Any, loss: str, outputs: Mapping[str, Tensor],
                 targets: Mapping[str, Any], indices: list[tuple[Tensor, Tensor]], **kwargs: Any):
        del kwargs
        if loss == "query_cgp":
            if "query_cgp_temporal_attention" not in outputs:
                zero = outputs["pred_logits"].sum() * 0.0
                return {"loss_query_cgp_bind": zero, "loss_query_cgp_route": zero}
            bind = _binding_loss(
                outputs,
                targets,
                indices,
                attention_key="query_cgp_temporal_attention",
                video_mask_key="query_cgp_video_mask",
                target_mode=self._causal_binding_target,
                span_loss_type=self.span_loss_type,
            )
            routes = outputs.get("query_cgp_basis_weights")
            if routes is None:
                route = bind * 0.0
            else:
                route = _matched_route_loss(routes, indices)
            return {"loss_query_cgp_bind": bind, "loss_query_cgp_route": route}
        if loss == "native_bind":
            key = "native_d1_temporal_attention"
            mask_key = "native_video_mask"
            if key not in outputs or mask_key not in outputs:
                zero = outputs["pred_logits"].sum() * 0.0
                return {"loss_native_bind": zero}
            bind = _binding_loss(
                outputs,
                targets,
                indices,
                attention_key=key,
                video_mask_key=mask_key,
                target_mode=self._causal_binding_target,
                span_loss_type=self.span_loss_type,
            )
            return {"loss_native_bind": bind}
        return original_get_loss(loss, outputs, targets, indices)

    criterion.get_loss = MethodType(get_loss, criterion)
    criterion._causal_loss_installed = True


def install_native_binding_capture(model: Any) -> None:
    """Capture differentiable D1 native cross-attention without changing state_dict."""

    decoder = model.transformer.decoder
    for layer in decoder.layers:
        attention = layer.multihead_attn
        if getattr(attention, "_causal_native_forward", False):
            continue
        original = attention.forward

        def wrapped(module: Any, *args: Any, _original=original, **kwargs: Any):
            call_kwargs = dict(kwargs)
            call_kwargs["need_weights"] = True
            call_kwargs["average_attn_weights"] = False
            output, weights = _original(*args, **call_kwargs)
            module._causal_last_attention = weights
            return output, weights

        attention.forward = MethodType(wrapped, attention)
        attention._causal_native_forward = True
        attention._causal_last_attention = None

    if getattr(model, "_causal_native_model_forward", False):
        return
    original_model_forward = model.forward

    def model_forward(module: Any, *args: Any, **kwargs: Any):
        outputs = original_model_forward(*args, **kwargs)
        src_vid_mask = kwargs.get("src_vid_mask")
        if src_vid_mask is None and len(args) >= 4:
            src_vid_mask = args[3]
        if src_vid_mask is None:
            raise ValueError("native binding capture requires src_vid_mask")
        layer_attention = module.transformer.decoder.layers[0].multihead_attn._causal_last_attention
        if layer_attention is None:
            raise RuntimeError("D1 native cross-attention was not captured")
        if layer_attention.ndim == 4:
            layer_attention = layer_attention.mean(dim=1)
        video_length = src_vid_mask.shape[1]
        native = layer_attention[..., :video_length]
        valid = src_vid_mask.bool().to(native.dtype).unsqueeze(1)
        native = native * valid
        native = native / native.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(native.dtype).eps)
        out = dict(outputs)
        out["native_d1_temporal_attention"] = native
        out["native_video_mask"] = src_vid_mask.bool()
        return out

    model.forward = MethodType(model_forward, model)
    model._causal_native_model_forward = True


def strip_query_cgp_state(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Return a checkpoint state with the entire ``query_cgp.*`` namespace removed."""

    return {key: value for key, value in state_dict.items() if not key.startswith("query_cgp.")}
