from __future__ import annotations

import numpy as np
import unittest

from causal_occurrence_lab.metrics import binding_metrics, fixed_k_metrics


class MetricsTest(unittest.TestCase):
    def test_duplicate_attribution_is_nonnegative_and_uses_one_gt_per_prediction(self):
        result = fixed_k_metrics(
            [[0.0, 2.0], [0.0, 2.0]],
            [[0.0, 2.0], [4.0, 6.0]],
            ks=(2,),
            thresholds=(0.5,),
        )
        self.assertEqual(result["duplicate_rate@2_05"], 0.5)
        self.assertGreaterEqual(result["duplicate_rate@2_05"], 0.0)
        self.assertLessEqual(result["duplicate_rate@2_05"], 1.0)
        self.assertEqual(result["unique_attributed_gt@2_05"], 1)


    def test_length_normalized_binding_can_reverse_a_long_window_bias(self):
        attention = np.asarray([[0.15, 0.15, 0.15, 0.15, 0.4]], dtype=np.float64)
        result = binding_metrics(
            attention,
            [[0.0, 8.0], [8.0, 10.0]],
            [0],
            [1],
            clip_length=2.0,
            duration=10.0,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["aec"], 0.0)
        self.assertEqual(result["aec_norm"], 1.0)
