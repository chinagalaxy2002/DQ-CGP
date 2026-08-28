# Causal occurrence-binding results

This directory contains artifacts produced by the isolated
`causal_occurrence_lab` harness. Existing source directories in the repository
were not modified.

## Phase 1: existing checkpoints

The `phase1_existing_checkpoints/` artifacts evaluate the existing baseline and
DQ-CGP checkpoints before any new causal-ablation training. The
multi-occurrence subset contains 160 qids. The main measured results are raw-
span diagnostic metrics (the formal `PostProcessorDETR` is not applied):

| Run | D1 mAP | D2 mAP | Coverage@5@0.5 | AEC-D1 final | AEC-D2 | ECR-D1 | ECR-D2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 6.39 | 6.26 | 0.1809 | 0.4897 | 0.4259 | 0.7752 | 0.8704 |
| dq_active | 14.99 | 15.19 | 0.4370 | 0.7849 | 0.8088 | 0.3148 | 0.3342 |
| dq_beta_zero | 14.99 | 15.47 | 0.4349 | 0.7818 | 0.8072 | 0.3210 | 0.3352 |
| dq_stripped | 14.99 | 15.47 | 0.4349 | 0.7818 | 0.8072 | 0.3210 | 0.3352 |

Complete per-qid records and D1/D2 submissions are kept under each run
directory. Corrected one-prediction-to-one-GT duplicate attribution is used,
so recorded DAR values remain in `[0, 1]`.

The active-vs-beta-zero diagnostics are in
`phase1_existing_checkpoints/tables/causal_comparisons.md`:

- mean classification probability difference: `0.00703`;
- mean span difference: `0.299` seconds;
- mean Top-5 query ranking Jaccard: `0.96879`;
- mean relative residual update: `0.05271`.

The 100-sample numerical equivalence test is in
`phase1_existing_checkpoints/strip_equivalence.json`; all four checked output
tensors have maximum absolute difference `0.0` at the fixed single-thread
verification setting.

The CLIP similarity stratification is in
`phase1_existing_checkpoints/tables/similarity_stratification.md`. It contains
the low/medium/high equal-sized tertiles (53/53/54 qids) and both baseline and
DQ-CGP metrics.

## Smoke logs

`smoke/supervision_only_seed2023_1batch/` and
`smoke/native_bind_seed2023_1batch/` contain one-batch training smoke logs,
validation logs, trajectory losses, and predictions. These are implementation
checks, not full causal-ablation results. The native-binding smoke evaluation is
under `smoke/native_bind_eval/`.

`twin_moment_smoke/` contains the generated three-example synthetic data smoke
test and its manifest. Full Twin-Moment evaluation was not claimed from this
smoke run.

## Reproduction

See `causal_occurrence_lab/README.md` for phase-1 analysis, first-round
training, multi-seed, trajectory, similarity, and Twin-Moment commands. The
formal model-selection rule remains validation `MR-full-mAP`; checkpoints are
intentionally excluded from this commit.
