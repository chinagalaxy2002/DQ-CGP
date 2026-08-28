from __future__ import annotations

import torch
import unittest

from causal_occurrence_lab.controls import install_query_cgp_controls, strip_query_cgp_state
from experiments.vmr_cgp.query_cgp import DETRQueryCGP


class ControlsTest(unittest.TestCase):
    def test_supervision_only_keeps_diagnostics_but_skips_injection(self):
        torch.manual_seed(3)
        module = DETRQueryCGP(hidden_dim=8, num_basis=4, prompt_length=2, router_hidden_dim=8, frf_hidden_dim=8, beta=0.05)
        install_query_cgp_controls(type("Model", (), {"query_cgp": module})(), inject_residual=False)
        decoder = torch.randn(3, 1, 8)
        memory = torch.randn(6, 1, 8)
        padding = torch.zeros(1, 6, dtype=torch.bool)
        semantic = torch.randn(1, 8)
        output = module(decoder, memory, padding, semantic, 4)
        self.assertTrue(torch.equal(output, decoder))
        self.assertIsNotNone(module.last_output)
        self.assertEqual(tuple(module.last_output.temporal_attention.shape), (1, 3, 4))


    def test_strip_removes_only_query_cgp_namespace(self):
        state = {"query_cgp.a": torch.ones(1), "class_embed.weight": torch.ones(2, 2)}
        stripped = strip_query_cgp_state(state)
        self.assertEqual(list(stripped), ["class_embed.weight"])
