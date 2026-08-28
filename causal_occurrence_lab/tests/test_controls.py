from __future__ import annotations

import torch
import unittest

from causal_occurrence_lab.controls import (
    install_criterion_controls,
    install_query_cgp_controls,
    strip_query_cgp_state,
)
from experiments.vmr_cgp.query_cgp import DETRQueryCGP
from models.moment_detr_gmr.moment_detr import SetCriterion


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

    @staticmethod
    def _criterion():
        return SetCriterion(
            matcher=None,
            weight_dict={},
            eos_coef=0.1,
            losses=[],
            span_loss_type="l1",
            max_v_l=8,
        )

    @staticmethod
    def _query_cgp_loss_inputs():
        torch.manual_seed(11)
        attention = torch.softmax(torch.randn(2, 3, 5, dtype=torch.float64), dim=-1)
        routes = torch.softmax(torch.randn(2, 3, 2, dtype=torch.float64), dim=-1)
        outputs = {
            "pred_logits": torch.zeros(2, 3, 2, dtype=torch.float64),
            "query_cgp_temporal_attention": attention,
            "query_cgp_basis_weights": routes,
            "query_cgp_video_mask": torch.ones(2, 5, dtype=torch.bool),
        }
        targets = {
            "span_labels": [
                {"spans": torch.tensor([[0.2, 0.2], [0.7, 0.2]], dtype=torch.float64)},
                {"spans": torch.tensor([[0.4, 0.2], [0.8, 0.1]], dtype=torch.float64)},
            ]
        }
        indices = [
            (torch.tensor([0, 2]), torch.tensor([0, 1])),
            (torch.tensor([1]), torch.tensor([1])),
        ]
        return outputs, targets, indices

    def test_causal_matched_binding_and_route_match_production(self):
        outputs, targets, indices = self._query_cgp_loss_inputs()
        production = self._criterion()
        causal = self._criterion()
        install_criterion_controls(causal, binding_target="matched")

        expected = production.get_loss("query_cgp", outputs, targets, indices)
        actual = causal.get_loss("query_cgp", outputs, targets, indices)
        for key in ("loss_query_cgp_bind", "loss_query_cgp_route"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0.0, atol=0.0)

    def test_route_loss_prefers_confident_globally_diverse_routes(self):
        attention = torch.ones(1, 2, 5, dtype=torch.float64) / 5
        targets = {
            "span_labels": [
                {"spans": torch.tensor([[0.2, 0.2], [0.7, 0.2]], dtype=torch.float64)}
            ]
        }
        indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
        collapsed_outputs = {
            "pred_logits": torch.zeros(1, 2, 2, dtype=torch.float64),
            "query_cgp_temporal_attention": attention,
            "query_cgp_basis_weights": torch.tensor(
                [[[1.0, 0.0], [1.0, 0.0]]], dtype=torch.float64
            ),
            "query_cgp_video_mask": torch.ones(1, 5, dtype=torch.bool),
        }
        diverse_outputs = {
            key: value.clone() for key, value in collapsed_outputs.items()
        }
        diverse_outputs["query_cgp_basis_weights"] = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]]],
            dtype=torch.float64,
        )
        production = self._criterion()
        diverse = production.get_loss("query_cgp", diverse_outputs, targets, indices)
        collapsed = production.get_loss("query_cgp", collapsed_outputs, targets, indices)
        torch.testing.assert_close(
            diverse["loss_query_cgp_route"],
            -torch.log(torch.tensor(2.0, dtype=torch.float64)),
            rtol=0.0,
            atol=1e-12,
        )
        self.assertLess(
            float(diverse["loss_query_cgp_route"]),
            float(collapsed["loss_query_cgp_route"]),
        )
