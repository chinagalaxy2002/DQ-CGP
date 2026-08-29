"""Runtime controls that leave the production DQ-CGP implementation untouched."""

from __future__ import annotations

from types import MethodType


def install_residual_injection_control(model, inject_residual: bool) -> None:
    """Compute the full DQ path but optionally return the native D1 state.

    Unlike production ``set_beta(0)``, this does not trigger the identity fast
    path. Temporal attention and routing remain in the graph, so the binding
    and route losses retain exactly their production semantics.
    """

    module = getattr(model, "query_cgp", None)
    if module is None:
        raise ValueError("Residual control requires a DQ-CGP model")
    if getattr(module, "_working_part_control_installed", False):
        module._working_part_inject = bool(inject_residual)
        return

    original_forward = module.forward

    def controlled_forward(self, decoder_state, *args, **kwargs):
        old_fast_path = self._beta_is_zero
        self._beta_is_zero = False
        try:
            adapted_state = original_forward(decoder_state, *args, **kwargs)
        finally:
            self._beta_is_zero = old_fast_path
        if self._working_part_inject:
            return adapted_state
        if self.last_output is not None and hasattr(self.last_output, "_replace"):
            self.last_output = self.last_output._replace(adapted_state=decoder_state)
        return decoder_state

    module.forward = MethodType(controlled_forward, module)
    module._working_part_inject = bool(inject_residual)
    module._working_part_original_forward = original_forward
    module._working_part_control_installed = True


def injection_is_enabled(model) -> bool:
    module = getattr(model, "query_cgp", None)
    if module is None:
        return False
    return bool(getattr(module, "_working_part_inject", True))
