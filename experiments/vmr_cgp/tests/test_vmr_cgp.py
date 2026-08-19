from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from experiments.vmr_cgp.checkpoint import (
    load_vmr_cgp_state_compat,
    restore_vmr_cgp_options,
)
from experiments.vmr_cgp.vmr_cgp import VMRCGP
from models.moment_detr_gmr.moment_detr import build_model
from training.moment_detr_gmr.config import BaseOptions


class VMRCGPModuleTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.module = VMRCGP(
            hidden_dim=16,
            num_basis=4,
            prompt_length=3,
            router_hidden_dim=24,
            temperature=0.8,
            alpha_init=0.05,
        )
        self.video = torch.randn(2, 5, 16)
        self.video_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
        self.text = torch.randn(2, 4, 16)
        self.text_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

    def test_shapes_probabilities_and_multi_evidence(self) -> None:
        with torch.no_grad():
            self.module.text_key.weight.zero_()
            self.module.video_key.weight.zero_()
        output = self.module(self.video, self.video_mask, self.text, self.text_mask)
        self.assertEqual(output.enhanced_text.shape, self.text.shape)
        self.assertEqual(output.frame_logits.shape, self.video_mask.shape)
        self.assertEqual(output.token_frame_logits.shape, (2, 4, 5))
        self.assertEqual(output.basis_weights.shape, (2, 4, 4))
        self.assertEqual(output.prompt_sequence.shape, (2, 4, 3, 16))
        torch.testing.assert_close(
            output.basis_weights.sum(dim=-1), torch.ones((2, 4))
        )
        # Sigmoid relevance does not normalize over time: three zero logits
        # produce total mass 1.5 rather than a softmax mass of one.
        valid_probability_sum = torch.sigmoid(
            output.token_frame_logits[0, 0, :3]
        ).sum()
        torch.testing.assert_close(valid_probability_sum, torch.tensor(1.5))

    def test_padding_invariance(self) -> None:
        self.module.eval()
        original = self.module(
            self.video, self.video_mask, self.text, self.text_mask
        )
        changed_video = self.video.clone()
        changed_text = self.text.clone()
        changed_video[self.video_mask == 0] = 1000 * torch.randn_like(
            changed_video[self.video_mask == 0]
        )
        changed_text[self.text_mask == 0] = 1000 * torch.randn_like(
            changed_text[self.text_mask == 0]
        )
        changed = self.module(
            changed_video, self.video_mask, changed_text, self.text_mask
        )
        torch.testing.assert_close(
            original.enhanced_text[self.text_mask.bool()],
            changed.enhanced_text[self.text_mask.bool()],
            rtol=0,
            atol=1e-6,
        )
        torch.testing.assert_close(
            original.frame_logits[self.video_mask.bool()],
            changed.frame_logits[self.video_mask.bool()],
            rtol=0,
            atol=1e-6,
        )

    def test_identity_and_prompt_only_update(self) -> None:
        self.module.eval()
        with torch.no_grad():
            self.module.alpha.zero_()
        identity = self.module(
            self.video, self.video_mask, self.text, self.text_mask
        )
        torch.testing.assert_close(identity.enhanced_text, self.text)

        with torch.no_grad():
            self.module.alpha.fill_(1.0)
            self.module.basis_prompts.zero_()
        no_prompt = self.module(
            self.video, self.video_mask, self.text, self.text_mask
        )
        torch.testing.assert_close(no_prompt.enhanced_text, self.text)

    def test_gradient_reaches_all_cgp_stages(self) -> None:
        output = self.module(
            self.video, self.video_mask, self.text, self.text_mask
        )
        loss = (
            output.enhanced_text[..., 0].sum()
            + output.frame_logits.mean()
            + output.basis_weights[..., 0].mean()
        )
        loss.backward()
        for name in (
            "text_key.weight",
            "video_key.weight",
            "router.0.weight",
            "basis_prompts",
            "prompt_value.weight",
            "feature_gate.2.weight",
            "alpha",
        ):
            parameter = dict(self.module.named_parameters())[name]
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0, name)

    def test_fixed_residual_cannot_collapse_through_alpha_or_gate(self) -> None:
        module = VMRCGP(
            hidden_dim=16,
            num_basis=4,
            prompt_length=3,
            router_hidden_dim=24,
            alpha_init=0.1,
            alpha_trainable=False,
            gate_floor=0.1,
        )
        self.assertFalse(module.alpha.requires_grad)
        with torch.no_grad():
            module.feature_gate[-1].weight.zero_()
            module.feature_gate[-1].bias.fill_(-100.0)
        output = module(self.video, self.video_mask, self.text, self.text_mask)
        valid_delta = (
            output.enhanced_text - self.text
        )[self.text_mask.bool()]
        self.assertGreater(float(valid_delta.detach().abs().sum()), 0.0)


class VMRCGPMomentDETRTest(unittest.TestCase):
    @staticmethod
    def _options(model_name: str):
        manager = BaseOptions(model_name, "soccer_gmr", "clip_slowfast")
        manager.parse()
        options = manager.option
        options.device = "cpu"
        return options

    def test_model_identity_at_zero_alpha_and_checkpoint_restore(self) -> None:
        torch.manual_seed(19)
        baseline, _ = build_model(self._options("moment_detr"))
        vmr_cgp, _ = build_model(self._options("moment_detr_vmr_cgp"))
        load_vmr_cgp_state_compat(
            vmr_cgp,
            baseline.state_dict(),
            allow_initialize_vmr_cgp=True,
        )
        with torch.no_grad():
            vmr_cgp.vmr_cgp.alpha.zero_()
        baseline.eval()
        vmr_cgp.eval()

        text = torch.randn(2, 7, 512)
        text_mask = torch.ones(2, 7)
        video = torch.randn(2, 8, 2818)
        video_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0]]
        )
        baseline_output = baseline(text, text_mask, video, video_mask)
        vmr_output = vmr_cgp(text, text_mask, video, video_mask)
        torch.testing.assert_close(
            vmr_output["pred_logits"], baseline_output["pred_logits"]
        )
        torch.testing.assert_close(
            vmr_output["pred_spans"], baseline_output["pred_spans"]
        )

        restored = SimpleNamespace(use_vmr_cgp=False)
        checkpoint = {"model": vmr_cgp.state_dict(), "opt": self._options("moment_detr_vmr_cgp")}
        self.assertTrue(restore_vmr_cgp_options(restored, checkpoint))
        self.assertTrue(restored.use_vmr_cgp)
        self.assertFalse(hasattr(restored, "use_query_attention_mask"))

    def test_full_forward_criterion_backward(self) -> None:
        model, criterion = build_model(self._options("moment_detr_vmr_cgp"))
        text = torch.randn(2, 6, 512)
        text_mask = torch.ones(2, 6)
        video = torch.randn(2, 9, 2818)
        video_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0]]
        )
        outputs = model(text, text_mask, video, video_mask)
        targets = {
            "span_labels": [
                {"spans": torch.tensor([[0.25, 0.2], [0.75, 0.2]])},
                {"spans": torch.zeros((0, 2))},
            ],
            "exist_label": torch.tensor([1.0, 0.0]),
        }
        losses = criterion(outputs, targets)
        self.assertIn("loss_vmr_cgp_rel", losses)
        self.assertIn("loss_vmr_cgp_route", losses)
        weighted = sum(
            losses[key] * criterion.weight_dict[key]
            for key in losses
            if key in criterion.weight_dict
        )
        self.assertTrue(torch.isfinite(weighted))
        weighted.backward()
        self.assertIsNotNone(model.vmr_cgp.basis_prompts.grad)
        self.assertTrue(torch.isfinite(model.vmr_cgp.basis_prompts.grad).all())

    def test_v2_options_fix_alpha_and_restore_gate_floor(self) -> None:
        model, _ = build_model(self._options("moment_detr_vmr_cgp_v2"))
        self.assertFalse(model.vmr_cgp.alpha.requires_grad)
        self.assertAlmostEqual(float(model.vmr_cgp.alpha), 0.1, places=6)
        self.assertEqual(model.vmr_cgp.gate_floor, 0.1)

        restored = SimpleNamespace(use_vmr_cgp=False)
        checkpoint = {
            "model": model.state_dict(),
            "opt": self._options("moment_detr_vmr_cgp_v2"),
        }
        self.assertTrue(restore_vmr_cgp_options(restored, checkpoint))
        self.assertFalse(restored.vmr_cgp_alpha_trainable)
        self.assertEqual(restored.vmr_cgp_gate_floor, 0.1)

    def test_v2_main_detr_loss_trains_rcg_bps_and_frf(self) -> None:
        model, criterion = build_model(self._options("moment_detr_vmr_cgp_v2"))
        text = torch.randn(2, 6, 512)
        text_mask = torch.ones(2, 6)
        video = torch.randn(2, 9, 2818)
        video_mask = torch.ones(2, 9)
        outputs = model(text, text_mask, video, video_mask)
        targets = {
            "span_labels": [
                {"spans": torch.tensor([[0.25, 0.2], [0.75, 0.2]])},
                {"spans": torch.tensor([[0.50, 0.3]])},
            ],
        }
        losses = criterion(outputs, targets)
        main_loss = sum(
            value * criterion.weight_dict[name]
            for name, value in losses.items()
            if name in criterion.weight_dict
            and not name.startswith("loss_vmr_cgp")
        )
        main_loss.backward()
        for name in (
            "text_key.weight",
            "video_key.weight",
            "router.0.weight",
            "basis_prompts",
            "prompt_value.weight",
            "feature_gate.2.weight",
        ):
            gradient = dict(model.vmr_cgp.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(float(gradient.abs().sum()), 0.0, name)


if __name__ == "__main__":
    unittest.main()
