"""Unit tests for LS-DQ-CGP + QAP, gradients, and counterfactual bypass."""

import sys
from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "training" / "moment_detr_gmr"
for path in (ROOT, TRAIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.moment_detr_gmr.moment_detr import build_model
from config import BaseOptions
from ls_dq_cgp_tap_lab.cgp_module import LateSemanticCGP
from ls_dq_cgp_tap_lab.ls_dq_cgp_model import LSDQCGPModel, install_ls_dq_cgp_loss


class TestLSDQCGP(unittest.TestCase):

    def test_cgp_module_shapes_and_bypass(self):
        cgp = LateSemanticCGP(hidden_dim=256, num_basis=16, prompt_length=6)
        bsz, num_queries, dim = 2, 10, 256
        v_q = torch.randn(bsz, num_queries, dim, requires_grad=True)
        e_static = torch.randn(bsz, dim)
        h_q = torch.randn(bsz, num_queries, dim)

        # Active forward
        out_active = cgp(v_q, e_static, h_q, static_bypass=False)
        self.assertEqual(out_active.pred_logits.shape, (bsz, num_queries, 2))
        self.assertEqual(out_active.adapted_semantic.shape, (bsz, num_queries, dim))
        self.assertEqual(out_active.basis_weights.shape, (bsz, num_queries, 16))
        self.assertEqual(out_active.prompt_sequence.shape, (bsz, num_queries, 6, dim))
        self.assertEqual(out_active.prompt_attention.shape, (bsz, num_queries, 6))
        torch.testing.assert_close(
            out_active.prompt_attention.sum(dim=-1),
            torch.ones(bsz, num_queries),
        )

        # Check that visual_context gradient is blocked (stop-gradient)
        loss = out_active.pred_logits.sum()
        loss.backward()
        self.assertIsNone(v_q.grad)

        # Static bypass forward
        out_bypass = cgp(v_q, e_static, h_q, static_bypass=True)
        self.assertEqual(out_bypass.pred_logits.shape, (bsz, num_queries, 2))
        # Active and static bypass should produce different logits
        diff = (out_active.pred_logits - out_bypass.pred_logits).abs().max().item()
        self.assertGreater(diff, 1e-4)

    def test_full_model_forward_and_loss(self):
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

        bsz = 2
        l_vid, l_txt = 30, 20
        d_vid = opt.v_feat_dim
        d_txt = opt.t_feat_dim
        src_vid = torch.randn(bsz, l_vid, d_vid)
        src_vid_mask = torch.ones(bsz, l_vid)
        src_vid_mask[0, 25:] = 0
        src_txt = torch.randn(bsz, l_txt, d_txt)
        src_txt_mask = torch.ones(bsz, l_txt)
        src_txt_mask[0, 15:] = 0

        outputs = model(
            src_txt=src_txt,
            src_txt_mask=src_txt_mask,
            src_vid=src_vid,
            src_vid_mask=src_vid_mask,
        )

        self.assertIn("pred_logits", outputs)
        self.assertIn("pred_spans", outputs)
        self.assertIn("pred_exist_logits", outputs)
        self.assertIn("prompt_attention", outputs)
        self.assertEqual(outputs["pred_logits"].shape, (bsz, 10, 2))
        self.assertEqual(outputs["pred_spans"].shape, (bsz, 10, 2))
        self.assertEqual(outputs["pred_exist_logits"].shape, (bsz,))
        self.assertEqual(outputs["prompt_attention"].shape, (bsz, 10, 6))

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
        self.assertGreater(
            (active_outputs["pred_logits"] - rolled_outputs["pred_logits"]).abs().max().item(),
            1e-4,
        )
        model.context_roll = False
        model.train()

        targets = {
            "exist_label": torch.tensor([1.0, 1.0]),
            "span_labels": [
                {"spans": torch.tensor([[0.2, 0.1], [0.5, 0.2]])},
                {"spans": torch.tensor([[0.7, 0.15]])},
            ]
        }

        losses = criterion(outputs, targets)
        self.assertIn("loss_native_bind", losses)
        self.assertIn("loss_label", losses)
        self.assertIn("loss_span", losses)
        self.assertIn("loss_exist", losses)

        total_loss = sum(losses[k] * criterion.weight_dict.get(k, 1.0) for k in losses)
        total_loss.backward()

        # Check that CGP module parameters receive gradients
        self.assertIsNotNone(model.cgp.router[0].weight.grad)
        self.assertIsNotNone(model.cgp.basis_prompts.grad)
        self.assertIsNotNone(model.cgp.prompt_query_proj.weight.grad)
        self.assertIsNotNone(model.cgp.prompt_key_proj.weight.grad)
        self.assertIsNotNone(model.cgp.prompt_value_proj.weight.grad)
        self.assertIsNotNone(model.cgp.frf[0].weight.grad)
        self.assertIsNotNone(model.cgp.logit_scale.grad)


if __name__ == "__main__":
    unittest.main()
