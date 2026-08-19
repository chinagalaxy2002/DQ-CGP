from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.temporal_cgp.checkpoint import (
    load_model_state_compat,
    restore_tcgp_options,
)
from experiments.temporal_cgp.ablation import install_tcgp_ablation
from experiments.temporal_cgp.temporal_cgp import TemporalCGP
from models.moment_detr_gmr.moment_detr import build_model
from training.moment_detr_gmr.config import BaseOptions
from training.moment_detr_gmr.dataset import StartEndDataset, start_end_collate


class TemporalCGPTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.module = TemporalCGP(
            hidden_dim=16,
            num_basis=4,
            prompt_length=2,
            router_hidden_dim=24,
            temperature=0.7,
            alpha_init=0.1,
        )
        self.video = torch.randn(2, 5, 16)
        self.video_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
        self.text = torch.randn(2, 4, 16)
        self.text_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

    def test_shapes_masks_and_probabilities(self) -> None:
        output = self.module(
            self.video, self.video_mask, self.text, self.text_mask
        )
        self.assertEqual(output.adapted_query.shape, (2, 16))
        self.assertEqual(output.prompt_sequence.shape, (2, 2, 16))
        self.assertEqual(output.coarse_attention.shape, (2, 5))
        self.assertEqual(output.basis_weights.shape, (2, 4))
        self.assertTrue(torch.isfinite(output.adapted_query).all())
        self.assertTrue(
            torch.equal(
                output.coarse_attention[self.video_mask == 0],
                torch.zeros_like(output.coarse_attention[self.video_mask == 0]),
            )
        )
        torch.testing.assert_close(
            output.coarse_attention.sum(dim=1) + output.null_attention,
            torch.ones(2),
        )
        torch.testing.assert_close(output.basis_weights.sum(dim=1), torch.ones(2))

    def test_padding_invariance(self) -> None:
        self.module.eval()
        original = self.module(
            self.video, self.video_mask, self.text, self.text_mask
        ).adapted_query
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
        ).adapted_query
        torch.testing.assert_close(original, changed, rtol=0, atol=1e-6)

    def test_null_candidate_handles_all_invalid_video(self) -> None:
        video_mask = torch.zeros_like(self.video_mask)
        output = self.module(self.video, video_mask, self.text, self.text_mask)
        torch.testing.assert_close(output.null_attention, torch.ones(2))
        torch.testing.assert_close(
            output.coarse_attention, torch.zeros_like(output.coarse_attention)
        )
        self.assertTrue(torch.isfinite(output.adapted_query).all())

    def test_gradient_reaches_router_basis_and_frf(self) -> None:
        output = self.module(
            self.video, self.video_mask, self.text, self.text_mask
        )
        # Squared mean after LayerNorm is almost constant.  A directional loss
        # provides a meaningful gradient-flow check for router/basis/FRF.
        loss = output.adapted_query[:, 0].sum()
        loss.backward()
        for name in (
            "basis_prompts",
            "router.0.weight",
            "update_mlp.0.weight",
            "feature_gate.weight",
            "alpha",
        ):
            parameter = dict(self.module.named_parameters())[name]
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 1e-10, name)


class QueryAttentionMaskTest(unittest.TestCase):
    def _dataset(self, root: Path, corrected: bool) -> StartEndDataset:
        label = root / "data.jsonl"
        label.write_text(
            json.dumps(
                {
                    "qid": 1,
                    "query": "Find all passes.",
                    "vid": "video.mp4",
                    "duration": 6,
                    "relevant_windows": [[0, 2]],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        text_dir = root / "text"
        clip_dir = root / "clip"
        slowfast_dir = root / "slowfast"
        text_dir.mkdir(exist_ok=True)
        clip_dir.mkdir(exist_ok=True)
        slowfast_dir.mkdir(exist_ok=True)

        rng = np.random.default_rng(3)
        feature = rng.normal(size=(77, 512)).astype(np.float32)
        mask = np.zeros(77, dtype=np.float32)
        mask[:5] = 1
        np.savez(
            text_dir / "qid1.npz",
            last_hidden_state=feature,
            attention_mask=mask,
        )
        np.savez(clip_dir / "video.npz", features=np.ones((3, 2), np.float32))
        np.savez(slowfast_dir / "video.npz", features=np.ones((3, 3), np.float32))
        return StartEndDataset(
            dset_name="soccer_gmr",
            domain=None,
            data_path=str(label),
            v_feat_dirs=[str(clip_dir), str(slowfast_dir)],
            q_feat_dir=str(text_dir),
            max_q_l=32,
            max_v_l=75,
            ctx_mode="video",
            clip_len=2,
            load_labels=True,
            keep_empty_gt=True,
            use_query_attention_mask=corrected,
        )

    def test_legacy_and_corrected_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = self._dataset(root, corrected=False)
            corrected = self._dataset(root, corrected=True)
            self.assertEqual(legacy[0]["model_inputs"]["query_feat"].shape, (32, 512))
            self.assertEqual(corrected[0]["model_inputs"]["query_feat"].shape, (5, 512))
            _, batch = start_end_collate([corrected[0]])
            torch.testing.assert_close(
                batch["query_feat"][1], torch.ones((1, 5))
            )


class MomentDETRTCGPSmokeTest(unittest.TestCase):
    @staticmethod
    def _options(model_name: str):
        manager = BaseOptions(model_name, "soccer_gmr", "clip_slowfast")
        manager.parse()
        options = manager.option
        options.device = "cpu"
        return options

    def test_full_forward_backward_and_checkpoint_warm_start(self) -> None:
        baseline, _ = build_model(self._options("moment_detr"))
        tcgp, _ = build_model(self._options("moment_detr_tcgp"))
        self.assertFalse(any(key.startswith("tcgp.") for key in baseline.state_dict()))
        self.assertTrue(any(key.startswith("tcgp.") for key in tcgp.state_dict()))
        load_model_state_compat(
            tcgp, baseline.state_dict(), allow_initialize_tcgp=True
        )

        src_txt = torch.randn(2, 6, 512)
        src_txt_mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]])
        src_vid = torch.randn(2, 7, 2818)
        src_vid_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0]])
        output = tcgp(src_txt, src_txt_mask, src_vid, src_vid_mask)
        self.assertEqual(output["pred_logits"].shape, (2, 10, 2))
        self.assertEqual(output["pred_spans"].shape, (2, 10, 2))
        self.assertEqual(output["pred_exist_logits"].shape, (2,))
        loss = output["pred_logits"].square().mean() + output["pred_spans"].mean()
        loss.backward()
        self.assertIsNotNone(tcgp.tcgp.basis_prompts.grad)
        self.assertTrue(torch.isfinite(tcgp.tcgp.basis_prompts.grad).all())

    def test_checkpoint_option_restore_and_state_only_fallback(self) -> None:
        tcgp, _ = build_model(self._options("moment_detr_tcgp"))
        state = tcgp.state_dict()

        state_only_options = SimpleNamespace(
            use_tcgp=False,
            use_query_attention_mask=False,
        )
        self.assertTrue(
            restore_tcgp_options(state_only_options, {"model": state})
        )
        self.assertTrue(state_only_options.use_tcgp)
        self.assertTrue(state_only_options.use_query_attention_mask)

        saved_options = self._options("moment_detr_tcgp")
        restored_options = self._options("moment_detr")
        self.assertTrue(
            restore_tcgp_options(
                restored_options,
                {"model": state, "opt": saved_options},
            )
        )
        rebuilt, _ = build_model(restored_options)
        load_model_state_compat(rebuilt, state)
        self.assertTrue(restored_options.use_query_attention_mask)

    def test_normalized_query_ablation_replaces_only_adapted_feature(self) -> None:
        tcgp, _ = build_model(self._options("moment_detr_tcgp"))
        tcgp.eval()
        video = torch.randn(2, 5, 2818)
        video_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
        text = torch.randn(2, 4, 512)
        text_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        projected_video = tcgp.input_vid_proj(video)
        projected_text = tcgp.input_txt_proj(text)
        expected_pool = (
            projected_text * text_mask.to(projected_text.dtype).unsqueeze(-1)
        ).sum(1) / text_mask.sum(1, keepdim=True)
        expected = tcgp.tcgp.output_norm(expected_pool)

        handle = install_tcgp_ablation(tcgp, "normalized_query")
        try:
            ablated = tcgp.tcgp(
                projected_video, video_mask, projected_text, text_mask
            )
        finally:
            handle.remove()
        torch.testing.assert_close(ablated.adapted_query, expected)


if __name__ == "__main__":
    unittest.main()
