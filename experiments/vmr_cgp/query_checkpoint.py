"""Checkpoint compatibility for the optional DETR-query CGP module."""

from __future__ import annotations

from typing import Any, Mapping


QUERY_CGP_OPTION_KEYS = (
    "use_query_cgp",
    "query_cgp_num_basis",
    "query_cgp_prompt_length",
    "query_cgp_router_hidden_dim",
    "query_cgp_frf_hidden_dim",
    "query_cgp_temperature",
    "query_cgp_beta",
    "query_cgp_binding_loss_coef",
    "query_cgp_route_loss_coef",
    "query_cgp_use_semantic_mask",
    "use_query_attention_mask",
)


def state_dict_has_query_cgp(state_dict: Mapping[str, Any]) -> bool:
    """Return whether a state dict contains the independent v3 namespace."""
    return any(key.startswith("query_cgp.") for key in state_dict)


def restore_query_cgp_options(opt: Any, checkpoint: Mapping[str, Any]) -> bool:
    """Restore v3 architecture, loss, and input-semantics options.

    The state prefix is the source of truth for enabling DQ-CGP.  State-only
    checkpoints fall back to the v3-defined private semantic mask while
    retaining the paper-original encoder attention-mask path.
    """
    state_dict = checkpoint.get("model", {})
    if not state_dict_has_query_cgp(state_dict):
        return False

    saved_opt = checkpoint.get("opt")
    restored_semantic_mask = False
    restored_encoder_mask = False
    if saved_opt is not None:
        for key in QUERY_CGP_OPTION_KEYS:
            if isinstance(saved_opt, Mapping):
                value = saved_opt.get(key)
            else:
                value = getattr(saved_opt, key, None)
            if value is not None:
                setattr(opt, key, value)
                if key == "query_cgp_use_semantic_mask":
                    restored_semantic_mask = True
                elif key == "use_query_attention_mask":
                    restored_encoder_mask = True

    opt.use_query_cgp = True
    if not restored_semantic_mask:
        opt.query_cgp_use_semantic_mask = True
    if not restored_encoder_mask:
        opt.use_query_attention_mask = False
    return True


def load_query_cgp_state_compat(
    model: Any,
    state_dict: Mapping[str, Any],
    *,
    allow_initialize_query_cgp: bool = False,
) -> None:
    """Load strictly except for an explicit baseline-to-v3 warm start.

    Warm start accepts only parameters missing from the ``query_cgp.*``
    namespace.  Unexpected v1/v2 ``vmr_cgp.*`` state or any missing shared
    Moment-DETR parameter remains an error.
    """
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    allowed_missing = (
        allow_initialize_query_cgp
        and missing
        and all(key.startswith("query_cgp.") for key in missing)
    )
    if unexpected or (missing and not allowed_missing):
        raise RuntimeError(
            "Checkpoint/model mismatch. "
            f"missing_keys={missing}, unexpected_keys={unexpected}"
        )
