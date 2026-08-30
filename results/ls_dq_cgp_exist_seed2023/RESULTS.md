# LS-DQ-CGP with Existence Head (Seed 2023)

This directory records the completed `ls_dq_cgp_exist` run on the Soccer-GMR Standard split. The model uses late-semantic query adaptation, native D1 binding supervision, and a dedicated GMR existence head trained with empty-ground-truth samples.

## Configuration

| Item | Value |
|---|---|
| Seed | 2023 |
| Learning rate | `5e-5` |
| Maximum epochs | 400 |
| Early-stopping patience | 50 |
| Native binding coefficient | 0.2 |
| Basis prompts | 16 |
| Prompt length | 6 |
| Existence head | enabled, max pooling, loss coefficient 1.0 |
| Best checkpoint | epoch 124 (checkpoint field is zero-based `123`) |
| Best validation mAP | 19.99 |

Training stopped after epoch 175 because validation mAP had not exceeded the epoch-124 best for 50 evaluations.

## Standard Test Results

| Metric | Baseline | Native Binding | DQ-CGPv3 | LS-DQ-CGP + Exist |
|---|---:|---:|---:|---:|
| mAP | 6.14 | 14.46 | 17.72 | **18.07** |
| mR@1 | 4.16 | 9.90 | 11.92 | **12.35** |
| mR@3 | 6.48 | 16.13 | **19.35** | 18.74 |
| mR@5 | 7.89 | 19.77 | 23.59 | **24.49** |
| mR+@3 | 0.56 | 1.64 | 5.56 | **5.83** |
| mR+@5 | 0.69 | 4.43 | **10.20** | 8.71 |
| mIoU@1 | 12.30 | 25.42 | 28.80 | **30.03** |
| mIoU@3 | 11.65 | 24.37 | 26.44 | **27.20** |
| mIoU@5 | 11.61 | 24.19 | 26.29 | **27.01** |
| mIoU+@3 | 1.52 | 6.80 | 8.77 | **9.89** |
| mIoU+@5 | 1.54 | 6.21 | 8.39 | **9.35** |
| AUROC | 71.87 | **76.69** | 76.23 | 75.83 |
| G-mIoU@1 | 5.67 | 30.07 | 32.23 | **32.39** |
| G-mIoU@3 | 2.44 | 24.66 | **26.81** | 26.17 |
| G-mIoU@5 | 1.49 | 22.72 | **24.86** | 24.03 |
| Best validation mAP | 7.09 | 19.55 | **20.80** | 19.99 |

All test rows above use the same 1,036-query Standard test set. The DQ-CGPv3 column is the internally consistent reproduced checkpoint result (`mAP=17.72`, `AUROC=76.23`), rather than mixing it with the released-checkpoint result (`mAP=15.51`, `AUROC=77.33`).

Compared with DQ-CGPv3, LS-DQ-CGP + Exist improves test mAP by 0.35, mR@5 by 0.90, mIoU@1 by 1.23, and mIoU+@5 by 0.96. It remains lower on mR+@5 by 1.49 and AUROC by 0.40. These are single-seed results, so the small mAP difference should not be interpreted as a statistically established improvement.

## Counterfactual Tests

All three modes use the same epoch-124 checkpoint and the same existence scores.

| Metric | Active | Static bypass | Context roll |
|---|---:|---:|---:|
| mAP | **18.07** | 11.53 | 17.16 |
| mR@1 | **12.35** | 4.54 | 11.22 |
| mR@3 | **18.74** | 12.31 | 17.90 |
| mR@5 | **24.49** | 18.03 | 23.87 |
| mR+@3 | **5.83** | 3.34 | 2.45 |
| mR+@5 | **8.71** | 5.94 | 7.51 |
| mIoU@1 | **30.03** | 13.61 | 25.79 |
| mIoU+@3 | **9.89** | 3.40 | 5.53 |
| mIoU+@5 | **9.35** | 3.44 | 5.16 |
| AUROC | 75.83 | 75.83 | 75.83 |

Active exceeds static bypass by 6.54 mAP, showing that candidate-specific adapted semantics materially affect ranking. Rolling the visual context across queries reduces mAP by 0.91 and mR+@3 by 3.38, providing a stricter check that the improvement depends on the assigned local context rather than only on added parameters.

## Files

- `experiment.json`: exact high-level run configuration.
- `train.log` and `val.log`: complete training and validation logs.
- `best_soccer_gmr_val_preds*.json*`: best-validation predictions and metrics.
- `test_active/`: normal late-semantic inference.
- `test_static_bypass/`: adapted text replaced by static text.
- `test_context_roll/`: each query receives another query's local visual context.
- [Google Drive checkpoint](https://drive.google.com/open?id=1_ekDDphGKkHm67ovNxz-Y6o-1MULpd8u): published `ls_dq_cgp_exist_best_epoch124.ckpt`.

The evaluation `result.json` uses `<repo>` as a portable placeholder for the original local repository path.
