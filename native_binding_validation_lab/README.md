# Native-binding validation

This isolated experiment asks whether DQ-CGP can be reduced to a training-only
matched binding regularizer.

1. `train_native_binding.py` trains plain Moment-DETR with zero added trainable
   parameters. A forward hook captures native D1 cross-attention, restricts and
   renormalizes it over valid video tokens, and applies the same final-Hungarian
   matched GT-mass loss used by DQ binding.
2. `verify_stripped.py` removes all `query_cgp.*` tensors from a binding-only
   checkpoint and checks plain Moment-DETR predictions against the no-injection
   DQ model.

No production model, trainer, matcher, dataset, or evaluator file is edited.

## Experimental Results & Logs

All logs, checkpoints metrics, and official test results are stored in [`results/component_attribution_seed2023/native_binding/`](../results/component_attribution_seed2023/native_binding/).

### Standard Test Performance (Seed 2023)

| Metric | Baseline | Native-Binding (Zero Extra Params) | Binding-Only | Full DQ-CGP |
|---|---:|---:|---:|---:|
| **Test mAP** | 6.14 | 14.46 | 14.91 | **17.72** |
| **G-mIoU@1** | 5.67 | 30.07 | **35.03** | 32.23 |
| **mR@5** | 7.89 | 19.77 | 20.67 | **23.59** |
| **mR+@5** | 0.69 | 4.43 | 4.45 | **10.20** |
| **mIoU@5** | 11.61 | 24.19 | 23.04 | **26.29** |
| **AUROC** | 71.87 | **76.69** | - | 76.23 |
| **Best Val mAP** | 7.09 (ep11) | 19.55 (ep81) | 19.04 (ep159) | **20.80 (ep112)** |

The table uses the matched retrained baseline and the internally consistent
reproduced Full DQ-CGP checkpoint (`mAP=17.72`, `AUROC=76.23`). The separately
released epoch-86 checkpoint has `mAP=15.51` and `AUROC=77.33`; its AUROC is
not combined with the reproduced checkpoint's retrieval metrics.

### Direct File Links

- **Training Log**: [`results/component_attribution_seed2023/native_binding/train.log`](../results/component_attribution_seed2023/native_binding/train.log)
- **Validation Log**: [`results/component_attribution_seed2023/native_binding/val.log`](../results/component_attribution_seed2023/native_binding/val.log)
- **Test Result Summary**: [`results/component_attribution_seed2023/native_binding/test/result.json`](../results/component_attribution_seed2023/native_binding/test/result.json)
- **Full Test Metrics**: [`results/component_attribution_seed2023/native_binding/test/metrics.json`](../results/component_attribution_seed2023/native_binding/test/metrics.json)
- **Stripped Equivalence Check**: [`results/component_attribution_seed2023/native_binding/binding_only_stripped_equivalence.json`](../results/component_attribution_seed2023/native_binding/binding_only_stripped_equivalence.json)
- **Full Component Attribution Report**: [`results/component_attribution_seed2023/RESULTS.md`](../results/component_attribution_seed2023/RESULTS.md)
