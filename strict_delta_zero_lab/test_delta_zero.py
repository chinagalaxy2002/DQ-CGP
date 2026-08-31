"""Formula-level regression test for strict delta-zero inference."""

import unittest

import torch
import torch.nn.functional as F

from strict_delta_zero_lab.strict_delta_zero_model import StrictDeltaZeroCGP


class TestStrictDeltaZero(unittest.TestCase):
    def test_exact_formula_and_preserved_matcher(self):
        torch.manual_seed(2023)
        module = StrictDeltaZeroCGP(
            hidden_dim=16,
            num_basis=4,
            prompt_length=2,
            router_hidden_dim=8,
            frf_hidden_dim=32,
        ).eval()
        visual_context = torch.randn(2, 3, 16)
        static_semantic = torch.randn(2, 16)
        query_states = torch.randn(2, 3, 16)

        with torch.no_grad():
            output = module(
                visual_context, static_semantic, query_states
            )
            expanded = static_semantic.unsqueeze(1).expand(-1, 3, -1)
            expected_adapted = module.frf_norm(expanded)
            h_metric = F.normalize(
                module.visual_proj(query_states), p=2, dim=-1
            )
            e_metric = F.normalize(
                module.semantic_proj(expected_adapted), p=2, dim=-1
            )
            expected_scores = (
                module.logit_scale * (h_metric * e_metric).sum(dim=-1)
                + module.logit_bias
            )

        self.assertTrue(
            torch.equal(output.adapted_semantic, expected_adapted)
        )
        self.assertTrue(torch.equal(output.semantic_scores, expected_scores))

        # Legacy bypass projects raw E_static, so it is not this intervention.
        legacy_metric = F.normalize(
            module.semantic_proj(expanded), p=2, dim=-1
        )
        self.assertGreater(
            (legacy_metric - e_metric).abs().max().item(), 1e-5
        )

    def test_training_gradient_isolates_zeroed_residual(self):
        torch.manual_seed(2023)
        module = StrictDeltaZeroCGP(
            hidden_dim=16,
            num_basis=4,
            prompt_length=2,
            router_hidden_dim=8,
            frf_hidden_dim=32,
        )
        output = module(
            torch.randn(2, 3, 16),
            torch.randn(2, 16),
            torch.randn(2, 3, 16),
        )
        output.pred_logits.square().mean().backward()

        self.assertIsNotNone(module.frf_norm.weight.grad)
        self.assertIsNotNone(module.semantic_proj.weight.grad)
        self.assertIsNotNone(module.visual_proj.weight.grad)
        self.assertIsNone(module.basis_prompts.grad)
        self.assertIsNone(module.router[0].weight.grad)
        self.assertIsNone(module.frf[0].weight.grad)
        self.assertIsNone(module.frf_v_proj.weight.grad)


if __name__ == "__main__":
    unittest.main()
