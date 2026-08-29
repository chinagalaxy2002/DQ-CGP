from types import SimpleNamespace

import torch

from experiments.vmr_cgp.query_cgp import DETRQueryCGP
from dq_cgp_working_part_lab.controls import install_residual_injection_control
from dq_cgp_working_part_lab.specs import get_spec


def _inputs():
    return dict(
        decoder_state=torch.randn(10, 2, 8),
        memory=torch.randn(7, 2, 8),
        memory_key_padding_mask=torch.zeros(2, 7, dtype=torch.bool),
        query_semantic=torch.randn(2, 8),
        video_length=5,
    )


def test_no_inject_is_identity_but_keeps_diagnostics():
    module = DETRQueryCGP(8, num_basis=4, prompt_length=2, router_hidden_dim=8, frf_hidden_dim=16, beta=0.05)
    model = SimpleNamespace(query_cgp=module)
    install_residual_injection_control(model, False)
    inputs = _inputs()
    output = module(**inputs)
    assert torch.equal(output, inputs["decoder_state"])
    assert module.last_output is not None
    assert module.last_output.temporal_attention.requires_grad


def test_full_control_changes_state():
    module = DETRQueryCGP(8, num_basis=4, prompt_length=2, router_hidden_dim=8, frf_hidden_dim=16, beta=0.05)
    model = SimpleNamespace(query_cgp=module)
    install_residual_injection_control(model, True)
    inputs = _inputs()
    assert not torch.equal(module(**inputs), inputs["decoder_state"])


def test_exact_no_inject_contract():
    spec = get_spec("no_inject")
    assert spec == {"dq": True, "bind": 0.2, "route": 0.01, "inject": False}
