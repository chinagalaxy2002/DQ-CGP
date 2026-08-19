from __future__ import annotations

import unittest

import torch

from experiments.vmr_cgp.query_checkpoint import load_query_cgp_state_compat
from experiments.vmr_cgp.query_cgp import DETRQueryCGP
from models.moment_detr_gmr.moment_detr import build_model
from training.moment_detr_gmr.config import BaseOptions


class DETRQueryCGPModuleTest(unittest.TestCase):
    """Contract tests for the candidate-wise DQ-CGP adapter.

    The transformer-facing tensors follow PyTorch Transformer layout while
    diagnostics use batch-first layout:

    * decoder state: ``[num_queries, batch, hidden_dim]``;
    * encoder memory: ``[sequence, batch, hidden_dim]``;
    * temporal attention: ``[batch, num_queries, video_length]``.
    """

    def setUp(self) -> None:
        torch.manual_seed(31)
        self.hidden_dim = 16
        self.num_queries = 3
        self.batch_size = 2
        self.video_length = 5
        self.sequence_length = 7
        self.module = self._make_module(beta=0.05)
        self.module.eval()

        self.decoder_state = torch.randn(
            self.num_queries, self.batch_size, self.hidden_dim
        )
        self.memory = torch.randn(
            self.sequence_length, self.batch_size, self.hidden_dim
        )
        # True denotes padding, matching Transformer key-padding-mask
        # semantics.  Only the first ``video_length`` positions may receive
        # temporal attention.
        self.memory_key_padding_mask = torch.tensor(
            [
                [False, False, False, True, True, False, True],
                [False, False, False, False, True, False, False],
            ]
        )
        self.query_semantic = torch.randn(self.batch_size, self.hidden_dim)

    def _make_module(self, beta: float) -> DETRQueryCGP:
        return DETRQueryCGP(
            hidden_dim=self.hidden_dim,
            num_basis=4,
            prompt_length=3,
            router_hidden_dim=24,
            beta=beta,
        )

    def _forward(
        self,
        module: DETRQueryCGP | None = None,
        *,
        decoder_state: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        module = self.module if module is None else module
        return module(
            decoder_state=(
                self.decoder_state if decoder_state is None else decoder_state
            ),
            memory=self.memory if memory is None else memory,
            memory_key_padding_mask=self.memory_key_padding_mask,
            query_semantic=self.query_semantic,
            video_length=self.video_length,
        )

    def test_shapes_and_probability_normalization(self) -> None:
        adapted = self._forward()
        output = self.module.last_output

        self.assertIsNotNone(output)
        self.assertEqual(adapted.shape, self.decoder_state.shape)
        self.assertEqual(output.adapted_state.shape, self.decoder_state.shape)
        self.assertEqual(
            output.temporal_logits.shape,
            (self.batch_size, self.num_queries, self.video_length),
        )
        self.assertEqual(
            output.temporal_attention.shape,
            (self.batch_size, self.num_queries, self.video_length),
        )
        self.assertEqual(
            output.temporal_context.shape,
            (self.batch_size, self.num_queries, self.hidden_dim),
        )
        self.assertEqual(
            output.basis_weights.shape,
            (self.batch_size, self.num_queries, 4),
        )
        self.assertEqual(
            output.prompt_sequence.shape,
            (self.batch_size, self.num_queries, 3, self.hidden_dim),
        )
        self.assertEqual(
            output.pooled_prompt.shape,
            (self.batch_size, self.num_queries, self.hidden_dim),
        )
        self.assertEqual(
            output.frf_feature.shape,
            (self.batch_size, self.num_queries, self.hidden_dim),
        )
        self.assertEqual(
            output.residual_update.shape,
            (self.batch_size, self.num_queries, self.hidden_dim),
        )
        torch.testing.assert_close(
            output.basis_weights.sum(dim=-1),
            torch.ones(self.batch_size, self.num_queries),
        )

    def test_temporal_attention_masks_padding_and_sums_over_valid_clips(self) -> None:
        self._forward()
        attention = self.module.last_output.temporal_attention
        video_padding = self.memory_key_padding_mask[:, : self.video_length]

        padded_values = attention.masked_select(video_padding.unsqueeze(1))
        torch.testing.assert_close(
            padded_values,
            torch.zeros_like(padded_values),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            attention.sum(dim=-1),
            torch.ones(self.batch_size, self.num_queries),
            rtol=1e-6,
            atol=1e-6,
        )

        # Padded video memory must not affect either valid attention or the
        # adapted decoder state.
        original_adapted = self._forward().detach().clone()
        original_attention = (
            self.module.last_output.temporal_attention.detach().clone()
        )
        changed_memory = self.memory.clone()
        for batch_index in range(self.batch_size):
            for time_index in range(self.video_length):
                if bool(video_padding[batch_index, time_index]):
                    changed_memory[time_index, batch_index] = (
                        1000.0 * torch.randn(self.hidden_dim)
                    )
        changed_adapted = self._forward(memory=changed_memory)
        changed_attention = self.module.last_output.temporal_attention
        torch.testing.assert_close(
            changed_adapted, original_adapted, rtol=0, atol=1e-6
        )
        torch.testing.assert_close(
            changed_attention, original_attention, rtol=0, atol=1e-6
        )

    def test_each_detr_query_conditions_its_own_context_and_routing(self) -> None:
        # Identical candidate states must initially yield identical temporal
        # binding and routing for the same video/query semantic.
        shared_state = torch.randn(1, 1, self.hidden_dim)
        identical_state = shared_state.expand(2, 1, self.hidden_dim).clone()
        memory = self.memory[:, :1].clone()
        mask = self.memory_key_padding_mask[:1].clone()
        semantic = self.query_semantic[:1].clone()

        self.module(
            decoder_state=identical_state,
            memory=memory,
            memory_key_padding_mask=mask,
            query_semantic=semantic,
            video_length=self.video_length,
        )
        identical_attention = self.module.last_output.temporal_attention
        identical_weights = self.module.last_output.basis_weights
        torch.testing.assert_close(
            identical_attention[:, 0], identical_attention[:, 1]
        )
        torch.testing.assert_close(identical_weights[:, 0], identical_weights[:, 1])

        # Changing only candidate 1 must leave candidate 0 unchanged while
        # producing a distinct candidate-specific temporal context and route.
        conditioned_state = identical_state.clone()
        conditioned_state[1, 0] = torch.linspace(
            -4.0, 4.0, self.hidden_dim
        )
        self.module(
            decoder_state=conditioned_state,
            memory=memory,
            memory_key_padding_mask=mask,
            query_semantic=semantic,
            video_length=self.video_length,
        )
        conditioned = self.module.last_output
        torch.testing.assert_close(
            conditioned.temporal_attention[:, 0],
            identical_attention[:, 0],
            rtol=0,
            atol=1e-6,
        )
        self.assertFalse(
            torch.allclose(
                conditioned.temporal_attention[:, 0],
                conditioned.temporal_attention[:, 1],
                rtol=0,
                atol=1e-7,
            )
        )
        self.assertFalse(
            torch.allclose(
                conditioned.basis_weights[:, 0],
                conditioned.basis_weights[:, 1],
                rtol=0,
                atol=1e-7,
            )
        )

    def test_zero_beta_is_exact_identity_and_skips_diagnostics(self) -> None:
        identity_module = self._make_module(beta=0.0)
        identity_module.eval()
        adapted = self._forward(module=identity_module)
        torch.testing.assert_close(
            adapted, self.decoder_state, rtol=0, atol=0
        )
        self.assertIsNone(identity_module.last_output)

    def test_reference_binding_loss_is_candidate_specific(self) -> None:
        """Document the matched-query/GT-window binding-loss construction.

        The production criterion may live with Moment-DETR's Hungarian
        matcher.  This reference test fixes the intended gather semantics:
        each matched DETR query is rewarded only for attention mass inside
        its own matched GT window, rather than a union shared by all queries.
        """

        attention = torch.tensor(
            [
                [
                    [0.60, 0.30, 0.10, 0.00, 0.00],
                    [0.05, 0.05, 0.10, 0.40, 0.40],
                ]
            ]
        )
        gt_clip_masks = torch.tensor(
            [
                [True, True, False, False, False],
                [False, False, False, True, True],
            ]
        )
        matched_query_indices = torch.tensor([0, 1])
        matched_gt_indices = torch.tensor([0, 1])

        matched_attention = attention[0, matched_query_indices]
        matched_masks = gt_clip_masks[matched_gt_indices]
        matched_mass = (matched_attention * matched_masks).sum(dim=-1)
        binding_loss = -matched_mass.clamp_min(1e-8).log().mean()
        expected = -torch.tensor([0.90, 0.80]).log().mean()
        torch.testing.assert_close(binding_loss, expected)

        swapped_masks = gt_clip_masks[matched_gt_indices.flip(0)]
        swapped_mass = (matched_attention * swapped_masks).sum(dim=-1)
        swapped_loss = -swapped_mass.clamp_min(1e-8).log().mean()
        self.assertGreater(float(swapped_loss), float(binding_loss))


class DETRQueryCGPMomentDETRTest(unittest.TestCase):
    @staticmethod
    def _options(model_name: str):
        manager = BaseOptions(model_name, "soccer_gmr", "clip_slowfast")
        manager.parse()
        options = manager.option
        options.device = "cpu"
        return options

    @staticmethod
    def _inputs():
        return {
            "src_txt": torch.randn(2, 8, 512),
            "src_txt_mask": torch.ones(2, 8),
            "src_txt_semantic_mask": torch.tensor(
                [[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]]
            ),
            "src_vid": torch.randn(2, 9, 2818),
            "src_vid_mask": torch.tensor(
                [[1, 1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0]]
            ),
        }

    def test_v3_configuration_and_beta_zero_baseline_equivalence(self) -> None:
        torch.manual_seed(41)
        baseline, _ = build_model(self._options("moment_detr"))
        query_cgp, _ = build_model(self._options("moment_detr_vmr_cgp_v3"))
        load_query_cgp_state_compat(
            query_cgp,
            baseline.state_dict(),
            allow_initialize_query_cgp=True,
        )
        query_cgp.query_cgp.set_beta(0.0)
        baseline.eval()
        query_cgp.eval()

        inputs = self._inputs()
        baseline_outputs = baseline(
            src_txt=inputs["src_txt"],
            src_txt_mask=inputs["src_txt_mask"],
            src_vid=inputs["src_vid"],
            src_vid_mask=inputs["src_vid_mask"],
        )
        query_outputs = query_cgp(**inputs)
        torch.testing.assert_close(
            query_outputs["pred_logits"], baseline_outputs["pred_logits"],
            rtol=0, atol=0,
        )
        torch.testing.assert_close(
            query_outputs["pred_spans"], baseline_outputs["pred_spans"],
            rtol=0, atol=0,
        )
        self.assertNotIn("query_cgp_temporal_attention", query_outputs)

    def test_full_forward_criterion_and_main_loss_gradients(self) -> None:
        torch.manual_seed(43)
        model, criterion = build_model(self._options("moment_detr_vmr_cgp_v3"))
        outputs = model(**self._inputs())
        self.assertEqual(outputs["query_cgp_temporal_attention"].shape, (2, 10, 9))
        self.assertEqual(outputs["query_cgp_basis_weights"].shape, (2, 10, 16))

        targets = {
            "span_labels": [
                {"spans": torch.tensor([[0.22, 0.20], [0.72, 0.18]])},
                {"spans": torch.tensor([[0.50, 0.25]])},
            ]
        }
        losses = criterion(outputs, targets)
        self.assertIn("loss_query_cgp_bind", losses)
        self.assertIn("loss_query_cgp_route", losses)
        total_loss = sum(
            value * criterion.weight_dict[name]
            for name, value in losses.items()
            if name in criterion.weight_dict
        )
        self.assertTrue(torch.isfinite(total_loss))
        total_loss.backward()

        parameters = dict(model.query_cgp.named_parameters())
        for name in (
            "candidate_projection.weight",
            "memory_key_projection.weight",
            "router.0.weight",
            "basis_prompts",
            "frf.0.weight",
            "residual_projection.weight",
        ):
            gradient = parameters[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(float(gradient.abs().sum()), 0.0, name)


if __name__ == "__main__":
    unittest.main()
