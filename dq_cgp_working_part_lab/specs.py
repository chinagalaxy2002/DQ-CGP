"""Frozen component-ablation matrix."""

VARIANTS = {
    "baseline": dict(dq=False, bind=0.0, route=0.0, inject=False),
    "full": dict(dq=True, bind=0.2, route=0.01, inject=True),
    "no_inject": dict(dq=True, bind=0.2, route=0.01, inject=False),
    "no_binding": dict(dq=True, bind=0.0, route=0.01, inject=True),
    "no_route": dict(dq=True, bind=0.2, route=0.0, inject=True),
    "injection_only": dict(dq=True, bind=0.0, route=0.0, inject=True),
    "binding_only": dict(dq=True, bind=0.2, route=0.0, inject=False),
    "route_only": dict(dq=True, bind=0.0, route=0.01, inject=False),
}


def get_spec(name: str):
    if name not in VARIANTS:
        raise ValueError(f"Unknown variant {name!r}; choose from {sorted(VARIANTS)}")
    return dict(VARIANTS[name])
