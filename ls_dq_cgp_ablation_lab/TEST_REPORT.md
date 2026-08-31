# LS-DQ-CGP Test 报告（Seed 2023）

测试日期：2026-08-31。所有结果均在 Soccer-GMR Standard Test 上计算，共 1,036 条查询（544 正例、492 负例）。本报告固定使用 Seed 2023，不进行多种子实验。

## 覆盖结论

本轮目标范围内的完整模型、反事实、四个 inference-only 消融、五个重训练消融均已完成 Test。本次重新运行了本地的 DQ-CGPv3 发布 checkpoint，结果与仓库已有 `results/test/metrics.json` 完全一致，并在新的输出目录生成了 1,036 条 prediction JSONL，完成独立核验。

| 组别 | Test 覆盖 | 结果位置 |
|---|---|---|
| Full LS-DQ-CGP + Exist（active） | 已完成 | `outputs/ls_dq_cgp_exist_seed2023/test_active/` |
| Full 反事实：`static_bypass`、`context_roll` | 已完成 | `outputs/ls_dq_cgp_exist_seed2023/test_{static_bypass,context_roll}/` |
| 重训练：RCG、BPS、FRF、完整路径消融 | 5/5 完成 | `outputs/ls_ablation_*_seed2023/test/` |
| inference-only：RCG、BPS、FRF | 4/4 完成 | `outputs/ls_inference_ablation_*_seed2023/` |
| Moment-DETR-GMR、早期 Native Binding | 已完成总体 Test | `results/component_attribution_seed2023/` |
| DQ-CGPv3 发布 checkpoint | 本次重测完成 | `outputs/dq_cgp_v3_seed2023/test_recheck/` |
| LS-DQ-CGP 无 Exist | 已完成 | `outputs/ls_dq_cgp_seed2023/test_active/` |

## 主要 baseline

| 方法 / checkpoint | mAP | mR@5 | mR+@5 | mIoU@1 | mIoU+@5 | AUROC | G-mIoU@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moment-DETR-GMR Baseline（归档） | 6.14 | 7.89 | 0.69 | 12.30 | 1.54 | 71.87 | 5.67 |
| Native Binding（归档） | 14.46 | 19.77 | 4.43 | 25.42 | 6.21 | 76.69 | 30.07 |
| DQ-CGPv3 发布 checkpoint（重测） | 15.51 | 22.45 | 7.16 | 25.84 | 6.93 | **77.33** | **43.25** |
| DQ-CGPv3 reproduced checkpoint（历史记录） | 17.72 | 23.59 | **10.20** | 28.80 | 8.39 | 76.23 | 32.23 |
| LS-DQ-CGP（无 Exist） | 16.65 | 21.78 | 8.44 | 25.97 | 7.12 | 75.04 | 15.40 |
| **LS-DQ-CGP + Exist** | **18.07** | **24.49** | 8.71 | **30.03** | **9.35** | 75.83 | 32.39 |

注意：DQ-CGPv3 的发布 checkpoint 与 reproduced checkpoint 不是同一个 checkpoint。发布版重测 mAP 为 15.51，历史 reproduced 结果为 17.72；二者不可拼接为单一方法分数。

## 本次 DQ-CGPv3 重测

命令：

```bash
DQ_PYTHON=/home/guoxiangyu/miniconda3/envs/GMR/bin/python \
DQ_GPU=0 \
DQ_OUTPUT=outputs/dq_cgp_v3_seed2023/test_recheck \
bash scripts/evaluate_dq_cgp_v3.sh
```

检查结果：预测与 GT 的共享 qid 为 1,036，缺失 qid 为 0。输出包括：

- `moment_detr_gmr_test_submission.jsonl`：1,036 条预测；
- `metrics.json`：官方 GMR 全量指标。

关键指标为 mAP 15.51、mR@1/3/5 = 9.65/16.91/22.45、mIoU@1/3/5 = 25.84/23.71/23.56、AUROC 77.33、G-mIoU@1/3/5 = 43.25/38.38/36.81。这与仓库已有 `results/test/metrics.json` 一致。

## Full 模型的反事实与消融

### 同 checkpoint inference-only

| 干预 | mAP | 相对 Full |
|---|---:|---:|
| Full active | 18.07 | - |
| `rcg_uniform` | 17.39 | -0.68 |
| `bps_query_mean` | 17.90 | -0.17 |
| `bps_zero` | 17.96 | -0.11 |
| `frf_remove` | 12.63 | **-5.44** |
| `static_bypass` | 11.53 | **-6.54** |
| `context_roll` | 17.16 | -0.91 |

Full checkpoint 对 FRF 和动态语义适配的依赖最强；RCG 与 BPS 的直接干预影响较小。该表测量的是已训练 checkpoint 的依赖性，不代表重新训练后的可实现性能。

### 重训练消融

| 模型 | mAP | 相对 Full |
|---|---:|---:|
| Full LS-DQ-CGP + Exist | 18.07 | - |
| `bps_zero` | 17.37 | -0.70 |
| `rcg_uniform` | 16.77 | -1.30 |
| `frf_remove` | 16.74 | -1.33 |
| `bps_query_mean` | 15.70 | -2.37 |
| `native_binding_exist_aligned` | 15.40 | -2.67 |

重训练后，`bps_query_mean` 和移除完整 late-semantic 路径的下降最大。`frf_remove` 在固定存在性阈值下的 G-mIoU 高于 Full，但 mAP 下降；这应解释为存在性校准/阈值效应，而不能据此得出 FRF 不重要的结论。

## 分桶复评：Full 与 DQ-CGPv3 发布 checkpoint

| 分桶 | 样本数 | Full mAP | DQ-CGPv3 mAP | `Full - DQ` |
|---|---:|---:|---:|---:|
| SportsMoments ≤3 秒 | 81 | 3.71 | 4.14 | -0.43 |
| SportsMoments 4--5 秒 | 72 | 26.30 | 26.79 | -0.49 |
| SportsMoments ≥6 秒 | 63 | 32.33 | 20.93 | **+11.40** |
| 单目标 | 384 | 18.27 | 15.30 | +2.97 |
| 多目标 | 160 | 17.58 | 16.02 | +1.56 |
| Early | 96 | 18.32 | 13.86 | +4.46 |
| Middle | 357 | 18.53 | 16.62 | +1.91 |
| Late | 91 | 15.98 | 12.88 | +3.10 |

在发布 checkpoint 对比下，Full 的主要优势集中于 SportsMoments 的 ≥6 秒目标，以及 Early/Middle/Late 各时间位置；Full 对 ≤5 秒目标没有优势。由于 SportsMoments 的长度与数据源、单目标属性相关，长度结论仅限该数据源内部。

## 结论与仍需注意的边界

1. 核心 LS-DQ-CGP 与受控消融测试已经完整覆盖；不需要额外训练或补测。
2. Full 的总体 mAP 最高，但不是所有单项指标都最高：DQ-CGPv3 的某些 checkpoint 在 AUROC、G-mIoU 或 mR+@5 上更高。
3. Full 的可信优势是候选定位排序、Top-5 检索、多目标 IoU，尤其是 SportsMoments 的 ≥6 秒目标；极短目标不是其优势区间。
4. 全局 AUROC 受数据源正负样本分布影响。存在性能力应优先查看 WorldCup 内部结果与校准阈值结果。
5. 本报告全部为单种子结果，结论应限定为 Seed 2023 与当前 Standard split。
