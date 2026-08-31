"""Regression tests for component intervention definitions."""

import unittest

import torch

from ls_dq_cgp_lab.cgp_module import LateSemanticCGP
from ls_dq_cgp_ablation_lab.ablation_model import AblationLateSemanticCGP


class TestAblationModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2023)
        self.inputs = (
            torch.randn(2, 5, 16),
            torch.randn(2, 16),
            torch.randn(2, 5, 16),
        )

    def make(self, variant):
        return AblationLateSemanticCGP(
            hidden_dim=16, num_basis=4, prompt_length=3,
            router_hidden_dim=8, frf_hidden_dim=32,
            variant=variant,
        ).eval()

    def test_full_is_exactly_production_equivalent(self):
        reference = LateSemanticCGP(
            hidden_dim=16, num_basis=4, prompt_length=3,
            router_hidden_dim=8, frf_hidden_dim=32,
        ).eval()
        controlled = self.make("full")
        controlled.load_state_dict(reference.state_dict())
        with torch.no_grad():
            expected = reference(*self.inputs)
            actual = controlled(*self.inputs)
        for expected_tensor, actual_tensor in zip(expected, actual):
            self.assertTrue(torch.equal(expected_tensor, actual_tensor))

    def test_rcg_uniform(self):
        module = self.make("rcg_uniform")
        output = module(*self.inputs)
        self.assertTrue(
            torch.equal(
                output.basis_weights,
                torch.full_like(output.basis_weights, 0.25),
            )
        )
        output.pred_logits.sum().backward()
        self.assertIsNone(module.router[0].weight.grad)
        self.assertIsNotNone(module.basis_prompts.grad)
        self.assertIsNotNone(module.frf[0].weight.grad)

    def test_bps_query_mean(self):
        module = self.make("bps_query_mean")
        output = module(*self.inputs)
        self.assertTrue(
            torch.equal(
                output.pooled_prompt,
                output.pooled_prompt[:, :1].expand_as(output.pooled_prompt),
            )
        )

    def test_bps_zero(self):
        module = self.make("bps_zero")
        output = module(*self.inputs)
        self.assertEqual(output.pooled_prompt.count_nonzero().item(), 0)
        output.pred_logits.sum().backward()
        self.assertIsNone(module.router[0].weight.grad)
        self.assertIsNone(module.basis_prompts.grad)
        self.assertIsNotNone(module.frf[0].weight.grad)

    def test_frf_remove_preserves_routed_prompt_residual(self):
        module = self.make("frf_remove")
        output = module(*self.inputs)
        expanded = self.inputs[1].unsqueeze(1).expand(-1, 5, -1)
        expected = module.frf_norm(expanded + output.pooled_prompt)
        self.assertTrue(torch.equal(output.adapted_semantic, expected))
        output.pred_logits.sum().backward()
        self.assertIsNone(module.frf[0].weight.grad)
        self.assertIsNone(module.frf_v_proj.weight.grad)
        self.assertIsNotNone(module.router[0].weight.grad)
        self.assertIsNotNone(module.basis_prompts.grad)


if __name__ == "__main__":
    unittest.main()

