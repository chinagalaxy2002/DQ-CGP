# DQ-CGP released-checkpoint test-time beta sweep

This report records an inference-only counterfactual sweep on the published
`checkpoints/dq_cgp_v3_best_epoch86.ckpt`. The checkpoint is loaded once and
only `model.query_cgp.beta` is changed; model weights, data, post-processing,
and the official GMR evaluator remain fixed.

## Provenance

- Split: Soccer-GMR Standard test, 1,036 queries.
- Seed recorded by checkpoint: 2023.
- Trained beta: `0.05`.
- Checkpoint SHA256:
  `b0f142fedbacea1077ecc82781d206b10ebd82c18cbcf43c44c256b44af4f0bf`.
- The `beta=0.05` prediction SHA256 is
  `f591e93264d637ee719d78ca9c984b57b1f5a5bcbe80c9f2b47cc17aeabe5088`,
  exactly matching `results/test/moment_detr_gmr_test_submission.jsonl`.

## Results

| beta | mAP | AUROC | G-mIoU@1 | mR@5 | mR+@5 |
|---:|---:|---:|---:|---:|---:|
| 0.00 | **15.85** | 77.26 | 42.84 | **22.80** | 6.88 |
| 0.05 (trained) | 15.51 | 77.33 | 43.25 | 22.45 | 7.16 |
| 0.10 | 15.22 | **77.34** | **43.76** | 22.52 | 6.62 |
| 0.20 | 14.65 | 77.29 | 43.39 | 20.91 | 6.37 |
| 0.50 | 13.55 | 76.83 | 41.54 | 19.86 | **7.67** |
| 1.00 | 11.09 | 75.90 | 37.95 | 15.79 | 7.26 |
| 1.50 | 9.99 | 75.41 | 36.46 | 14.64 | 7.08 |
| 2.00 | 9.54 | 75.24 | 35.71 | 13.29 | 5.43 |
| 2.50 | 9.24 | 75.22 | 35.62 | 12.57 | 5.19 |
| 3.00 | 8.85 | 75.18 | 35.33 | 12.44 | 4.71 |

The exact aggregate values are also stored in [`summary.csv`](summary.csv).

## Interpretation

For this released checkpoint, removing the residual injection (`beta=0`)
increases retrieval mAP from 15.51 to 15.85. Increasing beta beyond the
trained value progressively degrades mAP. The result therefore does not
support a positive contribution of the inter-layer residual to mAP at this
checkpoint.

This conclusion is metric-specific: `beta=0.05` slightly improves AUROC,
G-mIoU@1, and mR+@5 over `beta=0`, while their individual optima occur at
different beta values. This is a single-checkpoint inference intervention,
not a retraining study and not a multi-seed estimate.
