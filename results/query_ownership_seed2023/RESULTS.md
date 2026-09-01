# Decoder query-ownership diagnostic

This diagnostic independently applies Hungarian matching to decoder layer 1
(`aux_outputs[0]`, before DQ residual injection) and layer 2 (final output),
then measures whether each ground-truth moment retains the same DETR-query
owner. It is an analysis of query assignment stability, not an official GMR
quality metric.

## Configuration and provenance

- Split: Soccer-GMR Standard validation, 465 qids.
- Positive/empty-GT qids: 255/210; total GT moments: 365.
- Decoder layers/native queries: 2/10.
- Seed/batch size/workers: 2023/4/0.
- DQ checkpoint: reproduced Epoch 112 checkpoint, SHA256
  `82a7b956afb40052f57af723bd9028f743e0d72ab4af9a18e08f64c26362912f`.
- Baseline: independently trained local Moment-DETR checkpoint.
- Paired bootstrap: 2,000 resamples by qid, weighted by number of GT moments.
- All 465 qids occur in all three runs; no abnormal assignment was observed.

The reproduced DQ checkpoint is not the published Epoch 86 checkpoint and is
also not the matched baseline. The cross-checkpoint comparison below is
descriptive; only active versus beta-zero is a same-checkpoint intervention.

## Main results

| Method | Owner retention | Handoff | Direct switch | Drop | Sample-macro retention |
|---|---:|---:|---:|---:|---:|
| Baseline | 270/365 = **73.97%** | 26.03% | 7.95% | 18.08% | 78.24% |
| DQ-CGP active (`beta=0.05`) | 325/365 = **89.04%** | 10.96% | 3.56% | 7.40% | 90.82% |
| DQ-CGP beta-zero | 325/365 = **89.04%** | 10.96% | 3.56% | 7.40% | 91.01% |

| GT count | Samples / GTs | Baseline | DQ active | DQ beta-zero |
|---|---:|---:|---:|---:|
| K=1 | 165 / 165 | 84.85% | 93.33% | **93.94%** |
| K=2 | 72 / 144 | 68.06% | **87.50%** | 86.81% |
| K>=3 | 18 / 56 | 57.14% | **80.36%** | **80.36%** |

| Comparison with independently trained baseline | Delta retention | 95% paired-bootstrap CI |
|---|---:|---:|
| DQ active - Baseline | +15.07 percentage points | [+8.70, +20.98] |
| DQ beta-zero - Baseline | +15.07 percentage points | [+8.94, +21.07] |

The aggregate values are also stored in [`summary.csv`](summary.csv).

## Layer drift and same-checkpoint isolation

| Method/event | Mean center shift (s) | Mean inter-layer tIoU |
|---|---:|---:|
| Baseline / all matched | 6.915 | 0.411 |
| Baseline / retained | 5.432 | 0.496 |
| Baseline / handoff | 11.130 | 0.169 |
| DQ active / all matched | 1.592 | 0.718 |
| DQ active / retained | 1.208 | 0.736 |
| DQ active / handoff | 4.714 | 0.573 |
| DQ beta-zero / all matched | 1.561 | 0.727 |
| DQ beta-zero / retained | 1.165 | 0.747 |
| DQ beta-zero / handoff | 4.778 | 0.569 |

Within the reproduced DQ checkpoint, active and beta-zero have identical
layer-1 assignment maps over all 465 qids, as expected because injection
occurs afterwards. Their layer-2 assignment maps differ for only two qids,
and aggregate GT-micro retention remains 325/365 in both modes.

The DQ checkpoint has much higher retention than the independent baseline,
especially for multi-GT samples. However, the near-identical active and
beta-zero results show that this difference cannot be attributed solely to
the inference-time beta residual. Training history, other learned parameters,
checkpoint selection, and additional seeds are required for causal attribution.
