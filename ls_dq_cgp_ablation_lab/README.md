# LS-DQ-CGP Controlled Ablation Lab

This directory contains component-level ablations without modifying production files under `ls_dq_cgp_lab/`. Every semantic variant is shape- and checkpoint-compatible with Full LS-DQ-CGP and supports both inference-only intervention and Seed-2023 from-scratch training.

## Intervention matrix

| Variant | Intervention | Preserved path | Question answered |
|---|---|---|---|
| `full` | None | RCG, BPS, FRF, matcher | New-code reproduction control |
| `rcg_uniform` | `w_qk = 1/K` | BPS, FRF, matcher; visual context still enters FRF | Is learned candidate-specific routing useful? |
| `bps_query_mean` | Replace each candidate prompt by the mean prompt across candidates in the same sample | RCG is computed; FRF and matcher remain | Is candidate-specific BPS output useful beyond sample-level prompt content? |
| `bps_zero` | `P_q = 0` | FRF still receives `E_static` and projected local visual context | Does the routed prompt contribute beyond FRF's direct inputs? |
| `frf_remove` | Replace `FRF([P_q,E,V_q])` by identity residual `P_q` | RCG, BPS, `frf_norm`, matcher | Is learned multimodal fusion useful beyond routed prompts? |
| `native_binding_exist_aligned` | Plain Moment-DETR classifier; no late-semantic CGP or semantic matcher | D1 binding, existence head, saliency supervision | What is the contribution of the complete late-semantic prediction path? |

`bps_query_mean` deliberately averages across DETR candidates. Averaging basis prompts directly would produce the same tensor as uniform basis weights and would duplicate `rcg_uniform` rather than isolate BPS.

## Fair training configuration

All from-scratch controls use:

- Soccer-GMR Standard split and the identical CLIP/SlowFast/text features;
- Seed 2023 only;
- existence head enabled, including empty-GT training samples;
- `mr_only=False`, saliency labels enabled, and `lw_saliency=1`;
- native D1 binding coefficient 0.2;
- AdamW, learning rate `5e-5`, weight decay `1e-4`;
- batch size 8, at most 400 epochs, validation every epoch, patience 50;
- best-validation-mAP checkpoint followed by Standard Test evaluation.

## Two complementary protocols

Inference-only intervention uses the same trained Full checkpoint and measures whether that checkpoint relies on a component:

```bash
python ls_dq_cgp_ablation_lab/evaluate_semantic_ablation.py \
  --variant rcg_uniform \
  --checkpoint outputs/ls_dq_cgp_exist_seed2023/best.ckpt \
  --output outputs/inference_ablation_rcg_uniform \
  --protocol inference_only --split test --gpu 0
```

From-scratch training lets the remaining modules compensate and estimates the achievable performance without that component:

```bash
PYTHON=/path/to/python bash ls_dq_cgp_ablation_lab/run_experiment.sh rcg_uniform 0
```

The two results answer different causal questions and should be reported separately.

## Inference-only sanity check

The four semantic interventions were evaluated on the Standard Test split with the
same Full LS-DQ-CGP+Exist epoch-124 checkpoint. These numbers validate the code
paths and measure reliance of that particular trained model; they are not a
replacement for the from-scratch controls above.

| Variant | mAP | Delta mAP | mR@1 | mR@3 | mR@5 | mIoU+@3 | mIoU+@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `full` | 18.07 | - | 12.35 | 18.74 | 24.49 | 9.89 | 9.35 |
| `rcg_uniform` | 17.39 | -0.68 | 11.73 | 17.85 | 23.39 | 8.45 | 8.12 |
| `bps_query_mean` | 17.90 | -0.17 | 12.05 | 18.86 | 24.54 | 7.53 | 7.09 |
| `bps_zero` | 17.96 | -0.11 | 12.39 | 18.69 | 23.45 | 8.40 | 7.88 |
| `frf_remove` | 12.63 | -5.44 | 5.95 | 12.53 | 20.38 | 4.90 | 4.88 |

The checkpoint strongly relies on FRF and modestly relies on learned RCG routing.
The BPS interventions have little effect on aggregate mAP but produce clearer
drops on multi-moment IoU. Claims about each module's achievable contribution
must wait for the separately trained controls, because unchanged modules can
adapt during retraining.

## tmux launch

Launch one Seed-2023 training and automatic Test evaluation:

```bash
bash ls_dq_cgp_ablation_lab/launch_tmux.sh rcg_uniform 0
bash ls_dq_cgp_ablation_lab/launch_tmux.sh bps_query_mean 0
bash ls_dq_cgp_ablation_lab/launch_tmux.sh bps_zero 0
bash ls_dq_cgp_ablation_lab/launch_tmux.sh frf_remove 0
bash ls_dq_cgp_ablation_lab/launch_tmux.sh native_binding_exist_aligned 0
```

Do not launch several long runs on the same GPU simultaneously. Each session is named `ls_ablation_<variant>_seed2023`, writes a persistent log under `logs/`, and evaluates its best checkpoint automatically after early stopping.

## Tests

```bash
python -m unittest ls_dq_cgp_ablation_lab.test_ablation_model
```

The tests verify production equivalence of `full`, exact intervention formulas, expected gradient disconnections, and preservation of intended downstream paths.
