# Strict Delta-Zero Ablation

This directory adds an inference-only control without modifying any file under `ls_dq_cgp_lab/`.

The legacy `static_bypass` uses raw static semantics in the matcher:

```text
semantic_proj(E_static)
```

It therefore bypasses both the learned residual and `frf_norm`. Its Active-to-Bypass difference (`18.07 -> 11.53 mAP`) is a strong removal of the adapted semantic prediction path, but it is not an exact estimate of setting only `Delta E_q = 0`.

The strict intervention implemented here is:

```text
Delta E_q = 0
E_adapt^q = frf_norm(E_static + Delta E_q)
          = frf_norm(E_static)
```

The trained `frf_norm`, `semantic_proj`, `visual_proj`, cosine matcher, logit scale, and logit bias remain unchanged. The original checkpoint is loaded with no missing or unexpected parameters, and no retraining is required.

Run the Standard Test evaluation with:

```bash
python strict_delta_zero_lab/evaluate_delta_zero.py \
  --checkpoint outputs/ls_dq_cgp_exist_seed2023/best.ckpt \
  --output results/ls_dq_cgp_exist_seed2023/test_delta_zero \
  --split test \
  --gpu 0
```

Run the formula-level regression test with:

```bash
python -m unittest strict_delta_zero_lab.test_delta_zero
```

## Standard Test result (epoch 124, Seed 2023)

| Metric | Active | Strict `Delta E_q = 0` | Legacy Static Bypass |
|---|---:|---:|---:|
| mAP | **18.07** | 11.61 | 11.53 |
| mR@1 | **12.35** | 4.68 | 4.54 |
| mR@3 | **18.74** | 12.47 | 12.31 |
| mR@5 | **24.49** | 18.19 | 18.03 |
| mR+@3 | **5.83** | 3.09 | 3.34 |
| mR+@5 | **8.71** | 5.94 | 5.94 |
| mIoU@1 | **30.03** | 13.62 | 13.61 |
| mIoU+@3 | **9.89** | 3.69 | 3.40 |
| mIoU+@5 | **9.35** | 3.73 | 3.44 |
| AUROC | 75.83 | 75.83 | 75.83 |

The exact intervention changes mAP from `18.07` to `11.61`, a decrease of **6.46 mAP**. The legacy bypass is another 0.08 lower. Thus the earlier 6.54-point result was numerically close but described the stronger path-removal intervention; **6.46 mAP** is the result attributable to the strict zero-residual inference control. This remains an inference-time counterfactual on one trained checkpoint, not a multi-seed estimate of a training-time causal effect.

Full predictions and metrics are stored under `results/ls_dq_cgp_exist_seed2023/test_delta_zero/`.

## From-scratch architecture control

The inference-only control asks whether the trained active model depends on its residual. A separate from-scratch control asks how well the model can optimize when the residual is zero throughout training. It uses the same Seed 2023, existence head, native-binding coefficient, learning rate, maximum epochs, early stopping rule, and Standard train/validation splits as the active run.

Launch the complete training and best-checkpoint Test evaluation pipeline with:

```bash
PYTHON=/path/to/python bash strict_delta_zero_lab/run_train_delta_zero.sh 0
```

The default output directory is `outputs/ls_dq_cgp_delta_zero_train_seed2023/`. No original file under `ls_dq_cgp_lab/` is changed.

## From-scratch training result (Seed 2023)

The run completed with early stopping at epoch 122. The best validation score was
`MR-full-mAP = 18.62` (best checkpoint selected by validation mAP).

The resulting Standard Test metrics were:

| Metric | Value |
|---|---:|
| AUROC | 77.40 |
| Rej-F1@0.4 | 42.21 |
| Acc@0.4 | 61.68 |
| Rej-F1@0.6 | 72.98 |
| Acc@0.6 | 69.69 |
| G-mIoU@1 / @3 / @5 | 24.25 / 19.22 / 17.26 |
| mAP | 14.89 |
| mR@1 / @3 / @5 | 9.87 / 16.54 / 20.12 |
| mR+@1 / @3 / @5 | 0.00 / 4.91 / 8.52 |
| mIoU@1 / @3 / @5 | 25.76 / 23.72 / 23.56 |
| mIoU+@1 / @3 / @5 | 0.00 / 8.30 / 8.05 |

The complete run log is stored at `logs/ls_dq_cgp_delta_zero_train.log`.
