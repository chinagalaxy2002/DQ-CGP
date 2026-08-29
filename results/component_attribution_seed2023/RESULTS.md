# DQ-CGP component attribution (seed 2023)

This release records nine controlled experiments designed to identify which
part of DQ-CGP provides the observed gain. All runs use the Soccer-GMR Standard
split, the same training/evaluation pipeline, seed 2023, and the GMR existence
head. The experiment code is isolated from the production implementation.

## Experimental design

Eight runs form a complete `Binding × Route × Injection` factorial design:

| Variant | Matched binding loss | Route loss | Residual injection |
|---|---:|---:|---:|
| baseline | 0 | 0 | off (no DQ) |
| injection_only | 0 | 0 | on |
| route_only | 0 | 0.01 | off |
| no_binding | 0 | 0.01 | on |
| binding_only | 0.2 | 0 | off |
| no_route | 0.2 | 0 | on |
| no_inject | 0.2 | 0.01 | off |
| full | 0.2 | 0.01 | on |

`no_inject` still computes the DQ temporal/routing branch and its losses but
passes the native D1 decoder state into D2. This differs from merely setting
beta to zero in the production fast path.

The ninth run, `native_binding`, removes the DQ branch entirely. It hooks the
native D1 decoder cross-attention, renormalizes attention over valid video
tokens, and applies the same final-Hungarian matched GT-mass supervision. It
adds zero trainable parameters and retains the original existence head.

## Official Standard-test results

| Variant | Best val mAP | Epoch | Test mAP | G-mIoU@1 | mR@5 | mR+@5 | mIoU@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 7.09 | 11 | 6.14 | 5.67 | 7.89 | 0.69 | 11.61 |
| injection_only | 8.13 | 57 | 6.71 | 6.42 | 10.94 | 0.69 | 10.64 |
| route_only | 8.30 | 9 | 7.84 | 6.35 | 12.60 | 1.25 | 12.79 |
| no_binding | 8.51 | 7 | 8.18 | 6.00 | 14.12 | 0.72 | 12.17 |
| binding_only | 19.04 | 159 | 14.91 | 35.03 | 20.67 | 4.45 | 23.04 |
| no_route | 20.02 | 149 | 14.38 | 30.75 | 21.69 | 5.90 | 22.01 |
| no_inject | 16.76 | 77 | 12.94 | 27.21 | 17.33 | 6.81 | 22.66 |
| full | 20.80 | 112 | **17.72** | 32.23 | **23.59** | **10.20** | **26.29** |
| native_binding | 19.55 | 81 | 14.46 | 30.07 | 19.77 | 4.43 | 24.19 |

## Attribution on test mAP

| Comparison | Difference |
|---|---:|
| Full − baseline | +11.58 |
| Full − no-inject (conditional injection effect) | +4.78 |
| Full − no-binding (conditional binding effect) | +9.54 |
| Full − no-route (conditional route effect) | +3.34 |
| Injection-only − baseline | +0.57 |
| No-inject − baseline | +6.80 |
| Factorial main effect: binding | +7.77 |
| Factorial main effect: route | +1.14 |
| Factorial main effect: injection | +1.29 |

For this seed, matched temporal binding supervision is the dominant component.
Residual injection alone is weak (+0.57 mAP), but contributes conditionally in
the full system (+4.78 over `no_inject`), indicating interaction with the
supervised branch. Route regularization has the smallest averaged main effect.

`native_binding` reaches 14.46 mAP versus 14.91 for `binding_only`. Moreover,
stripping all 25 `query_cgp.*` tensors from the binding-only checkpoint produces
exactly identical plain Moment-DETR-GMR outputs on all 255 positive validation
samples (maximum absolute difference 0 for logits, spans, existence logits,
and saliency). This supports the narrower conclusion that most of the binding
gain can be realized as training-only supervision without a DQ inference
module. The full model remains best, so these results do not support deleting
DQ-CGP outright.

These are single-seed component-attribution results, not estimates of
cross-seed uncertainty.

## Artifacts and reproduction

- Factorial code: [`dq_cgp_working_part_lab/`](../../dq_cgp_working_part_lab/)
- Native-binding code: [`native_binding_validation_lab/`](../../native_binding_validation_lab/)
- Machine-readable factorial summary: [`factorial/component_summary.json`](factorial/component_summary.json)
- Stripped-model equivalence check: [`native_binding/binding_only_stripped_equivalence.json`](native_binding/binding_only_stripped_equivalence.json)
- Each run directory contains `experiment.json`, training/validation logs,
  best/latest validation metrics, and official test `metrics.json`/`result.json`.

To launch the eight factorial runs in tmux on two GPUs:

```bash
bash dq_cgp_working_part_lab/launch_tmux.sh
```

To train and finalize the native-binding validation:

```bash
python native_binding_validation_lab/train_native_binding.py
bash native_binding_validation_lab/finalize_when_ready.sh
```
