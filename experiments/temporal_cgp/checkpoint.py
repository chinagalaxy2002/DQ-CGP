"""Strict checkpoint compatibility helpers for the optional T-CGP module."""

from __future__ import annotations

from typing import Any, Mapping


TCGP_OPTION_KEYS = (
    "use_tcgp",
    "use_query_attention_mask",
    "tcgp_num_basis",
    "tcgp_prompt_length",
    "tcgp_router_hidden_dim",
    "tcgp_temperature",
    "tcgp_alpha_init",
)


def state_dict_has_tcgp(state_dict: Mapping[str, Any]) -> bool:
    return any(key.startswith("tcgp.") for key in state_dict)


def restore_tcgp_options(opt: Any, checkpoint: Mapping[str, Any]) -> bool:
    """Restore only architecture-relevant T-CGP options from a checkpoint."""
    state_dict = checkpoint.get("model", {})
    has_tcgp = state_dict_has_tcgp(state_dict)
    if not has_tcgp:
        return False

    saved_opt = checkpoint.get("opt")
    restored_query_mask = False
    if saved_opt is not None:
        for key in TCGP_OPTION_KEYS:
            if isinstance(saved_opt, Mapping):
                value = saved_opt.get(key)
            else:
                value = getattr(saved_opt, key, None)
            if value is not None:
                setattr(opt, key, value)
                if key == "use_query_attention_mask":
                    restored_query_mask = True
    opt.use_tcgp = True
    # A state-only T-CGP checkpoint has no saved input-semantics flag.  T-CGP
    # was defined and trained with the corrected CLIP mask, so do not silently
    # inherit the legacy ``false`` value that is present in base.yml.
    if not restored_query_mask:
        opt.use_query_attention_mask = True
    return True


def load_model_state_compat(
    model: Any,
    state_dict: Mapping[str, Any],
    *,
    allow_initialize_tcgp: bool = False,
) -> None:
    """Load strictly, except when warming a T-CGP model from a baseline.

    Blanket ``strict=False`` is deliberately avoided: only missing ``tcgp.*``
    parameters are accepted for the documented warm-start case.
    """
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    allowed_missing = (
        allow_initialize_tcgp
        and missing
        and all(key.startswith("tcgp.") for key in missing)
    )
    if unexpected or (missing and not allowed_missing):
        raise RuntimeError(
            "Checkpoint/model mismatch. "
            f"missing_keys={missing}, unexpected_keys={unexpected}"
        )
