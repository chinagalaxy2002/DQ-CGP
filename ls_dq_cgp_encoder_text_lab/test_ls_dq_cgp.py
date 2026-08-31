"""Tests for Encoder-Text LS-DQ-CGP and its two counterfactuals."""

import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "training" / "moment_detr_gmr"
for path in (ROOT, TRAIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import BaseOptions
from models.moment_detr_gmr.moment_detr import build_model
from ls_dq_cgp_encoder_text_lab.cgp_module import EncoderTextLateSemanticCGP
from ls_dq_cgp_encoder_text_lab.ls_dq_cgp_model import (
    LSDQCGPModel,
    install_ls_dq_cgp_loss,
)


class TestEncoderTextCGP(unittest.TestCase):
    def test_active_and_pre_encoder_condition_use_expected_semantics(self):
        torch.manual_seed(7)
        batch_size, num_queries, dim = 2, 10, 32
        cgp = EncoderTextLateSemanticCGP(
            hidden_dim=dim, num_basis=4, prompt_length=6
        )
        inputs = {
            "visual_context": torch.randn(batch_size, num_queries, dim),
            "static_semantic": torch.randn(batch_size, dim),
            "encoder_semantic": torch.randn(batch_size, dim),
            "query_states": torch.randn(batch_size, num_queries, dim),
        }

        active = cgp(**inputs)
        pre_encoder = cgp(**inputs, pre_encoder_condition=True)

        expected_encoder = inputs["encoder_semantic"].unsqueeze(1).expand(
            batch_size, num_queries, dim
        )
        expected_static = inputs["static_semantic"].unsqueeze(1).expand(
            batch_size, num_queries, dim
        )
        torch.testing.assert_close(active.condition_semantic, expected_encoder)
        torch.testing.assert_close(pre_encoder.condition_semantic, expected_static)
        self.assertEqual(active.basis_weights.shape, (batch_size, num_queries, 4))
        self.assertEqual(active.pooled_prompt.shape, (batch_size, num_queries, dim))
        self.assertEqual(active.pred_logits.shape, (batch_size, num_queries, 2))
        self.assertGreater(
            (active.pred_logits - pre_encoder.pred_logits).abs().max().item(),
            1e-5,
        )

    def test_visual_context_is_detached_but_both_text_paths_are_trainable(self):
        batch_size, num_queries, dim = 2, 5, 16
        cgp = EncoderTextLateSemanticCGP(
            hidden_dim=dim, num_basis=4, prompt_length=3
        )
        visual = torch.randn(batch_size, num_queries, dim, requires_grad=True)
        static = torch.randn(batch_size, dim, requires_grad=True)
        encoder = torch.randn(batch_size, dim, requires_grad=True)
        queries = torch.randn(batch_size, num_queries, dim, requires_grad=True)

        output = cgp(
            visual_context=visual,
            static_semantic=static,
            encoder_semantic=encoder,
            query_states=queries,
        )
        output.pred_logits.sum().backward()

        self.assertIsNone(visual.grad)
        self.assertIsNotNone(static.grad)
        self.assertIsNotNone(encoder.grad)
        self.assertIsNotNone(queries.grad)
        self.assertIsNotNone(cgp.router[0].weight.grad)
        self.assertIsNotNone(cgp.basis_prompts.grad)
        self.assertIsNotNone(cgp.frf[0].weight.grad)

    def test_rejects_invalid_encoder_semantic_shape(self):
        cgp = EncoderTextLateSemanticCGP(hidden_dim=8)
        with self.assertRaisesRegex(ValueError, "encoder_semantic"):
            cgp(
                visual_context=torch.randn(1, 2, 8),
                static_semantic=torch.randn(1, 8),
                encoder_semantic=torch.randn(1, 2, 8),
                query_states=torch.randn(1, 2, 8),
            )


class TestEncoderTextLSDQCGPModel(unittest.TestCase):
    def test_full_model_forward_counterfactuals_loss_and_gradients(self):
        manager = BaseOptions("moment_detr", "soccer_gmr", "clip_slowfast")
        manager.parse()
        opt = manager.option
        opt.device = "cpu"
        opt.num_queries = 10
        opt.enc_layers = 2
        opt.dec_layers = 2
        opt.use_exist_head = True
        opt.mr_only = False

        base_model, criterion = build_model(opt)
        model = LSDQCGPModel(base_model)
        install_ls_dq_cgp_loss(criterion, model, coefficient=0.2)

        self.assertFalse(any("selector" in name for name, _ in model.named_parameters()))

        batch_size, video_length, text_length = 2, 30, 20
        src_vid = torch.randn(batch_size, video_length, opt.v_feat_dim)
        src_vid_mask = torch.ones(batch_size, video_length)
        src_vid_mask[0, 25:] = 0
        src_txt = torch.randn(batch_size, text_length, opt.t_feat_dim)
        src_txt_mask = torch.ones(batch_size, text_length)
        src_txt_mask[0, 15:] = 0

        model_inputs = {
            "src_txt": src_txt,
            "src_txt_mask": src_txt_mask,
            "src_vid": src_vid,
            "src_vid_mask": src_vid_mask,
        }
        outputs = model(**model_inputs)

        self.assertEqual(outputs["pred_logits"].shape, (batch_size, 10, 2))
        self.assertEqual(outputs["pred_spans"].shape, (batch_size, 10, 2))
        self.assertEqual(outputs["pred_exist_logits"].shape, (batch_size,))
        self.assertEqual(outputs["static_semantic"].shape, (batch_size, 256))
        self.assertEqual(outputs["encoder_semantic"].shape, (batch_size, 256))
        self.assertEqual(outputs["condition_semantic"].shape, (batch_size, 10, 256))
        torch.testing.assert_close(
            outputs["condition_semantic"],
            outputs["encoder_semantic"].unsqueeze(1).expand(-1, 10, -1),
        )
        self.assertGreater(
            (outputs["static_semantic"] - outputs["encoder_semantic"])
            .abs()
            .max()
            .item(),
            1e-5,
        )

        model.eval()
        with torch.no_grad():
            active_outputs = model(**model_inputs)
            model.pre_encoder_condition = True
            pre_encoder_outputs = model(**model_inputs)
            model.pre_encoder_condition = False
            model.context_roll = True
            rolled_outputs = model(**model_inputs)
            model.context_roll = False

        torch.testing.assert_close(
            pre_encoder_outputs["condition_semantic"],
            pre_encoder_outputs["static_semantic"].unsqueeze(1).expand(-1, 10, -1),
        )
        self.assertGreater(
            (active_outputs["pred_logits"] - pre_encoder_outputs["pred_logits"])
            .abs()
            .max()
            .item(),
            1e-5,
        )
        self.assertGreater(
            (active_outputs["pred_logits"] - rolled_outputs["pred_logits"])
            .abs()
            .max()
            .item(),
            1e-5,
        )

        model.train()
        targets = {
            "exist_label": torch.tensor([1.0, 1.0]),
            "span_labels": [
                {"spans": torch.tensor([[0.2, 0.1], [0.5, 0.2]])},
                {"spans": torch.tensor([[0.7, 0.15]])},
            ],
        }
        losses = criterion(outputs, targets)
        for loss_name in ("loss_native_bind", "loss_label", "loss_span", "loss_exist"):
            self.assertIn(loss_name, losses)

        total_loss = sum(
            losses[key] * criterion.weight_dict.get(key, 1.0) for key in losses
        )
        total_loss.backward()

        self.assertIsNotNone(model.cgp.router[0].weight.grad)
        self.assertIsNotNone(model.cgp.basis_prompts.grad)
        self.assertIsNotNone(model.cgp.frf[0].weight.grad)
        self.assertIsNotNone(model.cgp.logit_scale.grad)


if __name__ == "__main__":
    unittest.main()
