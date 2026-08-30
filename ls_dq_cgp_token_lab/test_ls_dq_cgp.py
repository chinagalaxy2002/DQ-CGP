"""Tests for Token-Selective LS-DQ-CGP and its counterfactuals."""

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
from ls_dq_cgp_token_lab.cgp_module import TokenSelectiveLateSemanticCGP
from ls_dq_cgp_token_lab.ls_dq_cgp_model import (
    LSDQCGPModel,
    install_ls_dq_cgp_loss,
)


class TestTokenSelectiveCGP(unittest.TestCase):
    def test_token_attention_shape_and_mask(self):
        cgp = TokenSelectiveLateSemanticCGP(
            hidden_dim=256, num_basis=16, prompt_length=6
        )
        batch_size, num_queries, text_length, dim = 2, 10, 20, 256
        visual = torch.randn(batch_size, num_queries, dim)
        static = torch.randn(batch_size, dim)
        text = torch.randn(batch_size, text_length, dim)
        queries = torch.randn(batch_size, num_queries, dim)
        mask = torch.ones(batch_size, text_length, dtype=torch.bool)
        mask[0, 15:] = False

        output = cgp(
            visual_context=visual,
            static_semantic=static,
            text_tokens=text,
            text_mask=mask,
            query_states=queries,
        )

        self.assertEqual(output.token_attention.shape, (2, 10, 20))
        self.assertTrue(
            torch.allclose(
                output.token_attention[0, :, 15:],
                torch.zeros_like(output.token_attention[0, :, 15:]),
                atol=1e-7,
            )
        )
        self.assertTrue(
            torch.allclose(
                output.token_attention.sum(dim=-1),
                torch.ones(batch_size, num_queries),
                atol=1e-5,
            )
        )

    def test_visual_context_is_detached_and_selector_is_trainable(self):
        batch_size, num_queries, text_length, dim = 2, 10, 20, 256
        cgp = TokenSelectiveLateSemanticCGP(hidden_dim=dim)
        visual = torch.randn(batch_size, num_queries, dim, requires_grad=True)
        text = torch.randn(batch_size, text_length, dim, requires_grad=True)
        static = torch.randn(batch_size, dim, requires_grad=True)
        queries = torch.randn(batch_size, num_queries, dim, requires_grad=True)
        mask = torch.ones(batch_size, text_length, dtype=torch.bool)

        output = cgp(
            visual_context=visual,
            static_semantic=static,
            text_tokens=text,
            text_mask=mask,
            query_states=queries,
        )
        output.pred_logits.sum().backward()

        self.assertIsNone(visual.grad)
        self.assertIsNotNone(text.grad)
        self.assertIsNotNone(static.grad)
        self.assertIsNotNone(queries.grad)
        self.assertIsNotNone(cgp.selector_visual_proj.weight.grad)
        self.assertIsNotNone(cgp.selector_text_proj.weight.grad)

    def test_active_token_static_and_static_bypass_differ(self):
        batch_size, num_queries, text_length, dim = 2, 10, 20, 256
        cgp = TokenSelectiveLateSemanticCGP(hidden_dim=dim)
        inputs = {
            "visual_context": torch.randn(batch_size, num_queries, dim),
            "static_semantic": torch.randn(batch_size, dim),
            "text_tokens": torch.randn(batch_size, text_length, dim),
            "text_mask": torch.ones(batch_size, text_length, dtype=torch.bool),
            "query_states": torch.randn(batch_size, num_queries, dim),
        }

        active = cgp(**inputs)
        token_static = cgp(**inputs, token_static_bypass=True)
        static_bypass = cgp(**inputs, static_bypass=True)

        self.assertGreater(
            (active.pred_logits - token_static.pred_logits).abs().max().item(),
            1e-5,
        )
        self.assertGreater(
            (active.pred_logits - static_bypass.pred_logits).abs().max().item(),
            1e-5,
        )

    def test_rejects_empty_text_mask(self):
        cgp = TokenSelectiveLateSemanticCGP(hidden_dim=8)
        with self.assertRaisesRegex(ValueError, "at least one valid text token"):
            cgp(
                visual_context=torch.randn(1, 2, 8),
                static_semantic=torch.randn(1, 8),
                text_tokens=torch.randn(1, 3, 8),
                text_mask=torch.zeros(1, 3, dtype=torch.bool),
                query_states=torch.randn(1, 2, 8),
            )


class TestTokenLSDQCGPModel(unittest.TestCase):
    def test_full_model_forward_loss_masks_and_gradients(self):
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

        batch_size, video_length, text_length = 2, 30, 20
        src_vid = torch.randn(batch_size, video_length, opt.v_feat_dim)
        src_vid_mask = torch.ones(batch_size, video_length)
        src_vid_mask[0, 25:] = 0
        src_txt = torch.randn(batch_size, text_length, opt.t_feat_dim)
        src_txt_mask = torch.ones(batch_size, text_length)
        src_txt_mask[0, 15:] = 0

        outputs = model(
            src_txt=src_txt,
            src_txt_mask=src_txt_mask,
            src_vid=src_vid,
            src_vid_mask=src_vid_mask,
        )

        self.assertEqual(outputs["pred_logits"].shape, (batch_size, 10, 2))
        self.assertEqual(outputs["pred_spans"].shape, (batch_size, 10, 2))
        self.assertEqual(outputs["pred_exist_logits"].shape, (batch_size,))
        self.assertEqual(
            outputs["token_attention"].shape, (batch_size, 10, text_length)
        )
        # SOT/EOT are excluded from local selection; padding is masked too.
        self.assertTrue(torch.equal(outputs["token_attention"][0, :, 0], torch.zeros(10)))
        self.assertTrue(torch.equal(outputs["token_attention"][0, :, 14], torch.zeros(10)))
        self.assertTrue(
            torch.equal(
                outputs["token_attention"][0, :, 15:], torch.zeros(10, 5)
            )
        )
        # Sample 1 fills the complete max-length window. Its final valid token
        # may be truncated content rather than EOT and must remain selectable.
        self.assertTrue(
            torch.equal(outputs["token_attention"][1, :, 0], torch.zeros(10))
        )
        self.assertTrue(torch.all(outputs["token_attention"][1, :, -1] > 0))

        model.eval()
        with torch.no_grad():
            active_outputs = model(
                src_txt=src_txt,
                src_txt_mask=src_txt_mask,
                src_vid=src_vid,
                src_vid_mask=src_vid_mask,
            )
            model.context_roll = True
            rolled_outputs = model(
                src_txt=src_txt,
                src_txt_mask=src_txt_mask,
                src_vid=src_vid,
                src_vid_mask=src_vid_mask,
            )
            model.context_roll = False
            model.token_static_bypass = True
            token_static_outputs = model(
                src_txt=src_txt,
                src_txt_mask=src_txt_mask,
                src_vid=src_vid,
                src_vid_mask=src_vid_mask,
            )
        self.assertGreater(
            (active_outputs["pred_logits"] - rolled_outputs["pred_logits"])
            .abs()
            .max()
            .item(),
            1e-5,
        )
        self.assertGreater(
            (active_outputs["pred_logits"] - token_static_outputs["pred_logits"])
            .abs()
            .max()
            .item(),
            1e-5,
        )
        model.token_static_bypass = False
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

        self.assertIsNotNone(model.cgp.selector_visual_proj.weight.grad)
        self.assertIsNotNone(model.cgp.selector_text_proj.weight.grad)
        self.assertIsNotNone(model.cgp.router[0].weight.grad)
        self.assertIsNotNone(model.cgp.basis_prompts.grad)
        self.assertIsNotNone(model.cgp.frf[0].weight.grad)
        self.assertIsNotNone(model.cgp.logit_scale.grad)


if __name__ == "__main__":
    unittest.main()
