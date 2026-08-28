from __future__ import annotations

import unittest

import torch

from occurrence_binding.context_roll import install_context_roll
from experiments.vmr_cgp.query_cgp import DETRQueryCGP


class ContextRollTest(unittest.TestCase):
    def test_context_roll_preserves_update_multiset_and_changes_pairing(self) -> None:
        torch.manual_seed(7)
        kwargs = dict(
            hidden_dim=16,
            num_basis=4,
            prompt_length=3,
            router_hidden_dim=24,
            frf_hidden_dim=32,
            beta=0.05,
        )
        active = DETRQueryCGP(**kwargs).eval()
        rolled = DETRQueryCGP(**kwargs).eval()
        rolled.load_state_dict(active.state_dict())
        install_context_roll(rolled)

        decoder_state = torch.randn(5, 2, 16)
        memory = torch.randn(8, 2, 16)
        padding = torch.zeros(2, 8, dtype=torch.bool)
        semantic = torch.randn(2, 16)
        call = dict(
            decoder_state=decoder_state,
            memory=memory,
            memory_key_padding_mask=padding,
            query_semantic=semantic,
            video_length=5,
        )
        active_state = active(**call)
        rolled_state = rolled(**call)
        active_output = active.last_output
        rolled_output = rolled.last_output
        assert active_output is not None
        assert rolled_output is not None

        torch.testing.assert_close(
            rolled_output.temporal_attention,
            active_output.temporal_attention,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            rolled_output.temporal_context,
            active_output.temporal_context.roll(1, dims=1),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            rolled_output.residual_update,
            active_output.residual_update.roll(1, dims=1),
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            rolled_output.residual_update.norm(dim=-1),
            active_output.residual_update.norm(dim=-1).roll(1, dims=1),
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertFalse(torch.allclose(active_state, rolled_state))


if __name__ == "__main__":
    unittest.main()

