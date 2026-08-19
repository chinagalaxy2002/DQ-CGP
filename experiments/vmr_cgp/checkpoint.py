"""Checkpoint compatibility for the optional VMR-CGP module."""

from __future__ import annotations

from typing import Any, Mapping


VMR_CGP_OPTION_KEYS = (
    "use_vmr_cgp",
    "vmr_cgp_num_basis",
    "vmr_cgp_prompt_length",
    "vmr_cgp_router_hidden_dim",
    "vmr_cgp_temperature",
    "vmr_cgp_alpha_init",
    "vmr_cgp_alpha_trainable",
    "vmr_cgp_gate_floor",
    "vmr_cgp_temporal_loss_coef",
    "vmr_cgp_route_loss_coef",
)


def state_dict_has_vmr_cgp(state_dict: Mapping[str, Any]) -> bool:
    return any(key.startswith("vmr_cgp.") for key in state_dict)


def restore_vmr_cgp_options(opt: Any, checkpoint: Mapping[str, Any]) -> bool:
    """Restore architecture/loss options without changing input semantics."""
    state_dict = checkpoint.get("model", {})
    if not state_dict_has_vmr_cgp(state_dict):
        return False

    saved_opt = checkpoint.get("opt")
    if saved_opt is not None:
        for key in VMR_CGP_OPTION_KEYS:
            if isinstance(saved_opt, Mapping):
                value = saved_opt.get(key)
            else:
                value = getattr(saved_opt, key, None)
            if value is not None:
                setattr(opt, key, value)
    opt.use_vmr_cgp = True
    return True


def load_vmr_cgp_state_compat(
    model: Any,
    state_dict: Mapping[str, Any],
    *,
    allow_initialize_vmr_cgp: bool = False,
) -> None:
    """Load strictly, except for an explicit baseline-to-VMR-CGP warm start."""
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    allowed_missing = (
        allow_initialize_vmr_cgp
        and missing
        and all(key.startswith("vmr_cgp.") for key in missing)
    )
    if unexpected or (missing and not allowed_missing):
        raise RuntimeError(
            "Checkpoint/model mismatch. "
            f"missing_keys={missing}, unexpected_keys={unexpected}"
        )
