# DQ-CGP: DETR-Query Compositional Generalization Prompter for GMR

本仓库提供 **DQ-CGP V3** 与 **LS-DQ-CGP (Late-Semantic DQ-CGP)** 的完整训练、推理、评测代码以及训练好的 checkpoint。当前推荐版本进一步加入独立的 **GMR existence head**，并在包含空 GT query 的训练集上联合优化。DQ-CGP 面向 Generalized Moment Retrieval（GMR）中的多窗口检索：它不再为所有 DETR candidates 生成同一个全局增强文本，而是把每个原生 DETR query 看作一个候选实例，为其生成独立的 temporal context、basis routing 和 adapted feature。

代码基于 Moment-DETR-GMR，并使用 Soccer-GMR Standard split。

---

## 1. 方法概览

### 1.1 LS-DQ-CGP: Late-Semantic 架构 (推荐)

在深入分析 APT (Adaptive Prompt Tuning) 的 Information Bottleneck 机制后，**LS-DQ-CGP** 彻底解耦了“时序定位”与“语义打分”：

```text
                     Native Binding Loss (0.2)
                            │
                            ▼
Video/Text → Encoder → D1 native attention
                            │
                            ▼
                      local V_query
                            │
                      stop-gradient
                            │
          E_static ─────────┤
                            ▼
                      RCG → BPS → FRF
                            │
                            ▼
                      E_adapt_query
                            │
                            │ Cosine Matching
                            ▼
D2 final query ─────────→ pred_logits (Relevance Score)
      │
      └─────────────────→ span head (Temporal Boundaries)
```

* **时序与语义解耦**：Moment-DETR Decoder 主干专心回归时间边界，不再被中间层微小残差所干扰。
* **视觉与监督同源**：通过 Native Binding 约束 D1 Cross-Attention 抓取精准的局部视频上下文 $V_q$，并用 `stop-gradient` 阻断梯度捷径。
* **动态语义直接主导检索**：由 RCG $\rightarrow$ BPS $\rightarrow$ FRF 生成的 $E_{adapt}^q$ 直接与 D2 隐状态 $h_q$ 进行余弦相似度打分输出 `pred_logits`。
* **显式空查询判别**：existence head 独立预测 query 是否存在相关时刻，使 GMR rejection 不再依赖检索分数的间接代理。

### 1.2 统一 Baseline 与方法谱系

本文档中的受控实验统一以 **Matched retrained Moment-DETR-GMR Baseline** 为根节点：相同 Soccer-GMR Standard split、Seed 2023、特征、训练/验证划分和官方评测器，并在需要公平比较 GMR classification 时使用相同 existence-head 配置。仓库中的旧论文 baseline（mAP 5.34）来自另一历史 checkpoint，仅作发布结果追溯，不用于计算下表的方法增益。

```text
Matched Moment-DETR-GMR Baseline
├── DQ-CGP: candidate temporal context + RCG/BPS/FRF
│   ├── Binding / Route / Injection 八组因子实验
│   ├── released checkpoint 与 reproduced checkpoint
│   └── beta sweep、query ownership、occurrence-binding diagnostics
└── Native Binding: 原生 D1 attention + matched binding loss
    └── LS-DQ-CGP: late-semantic matcher 直接生成 relevance logits
        ├── + Existence Head（当前推荐）
        ├── QAP（attentive prompt pooling）
        ├── Token-V1 / Token-V2（token selector）
        ├── Encoder-Text（encoder text mean，无 selector）
        └── RCG / BPS / FRF / strict delta-zero 消融
```

所有名称均表示相对其父节点增加或替换的模块。独立训练方法、从头消融和推理反事实在下一节分表报告，避免将不同 checkpoint 或不同实验协议放在同一因果比较中。

---

## 2. 已发布 Checkpoint 与官方评测结果

### Checkpoints
* **LS-DQ-CGP + Existence Head（推荐）**: [Google Drive 下载](https://drive.google.com/open?id=1_ekDDphGKkHm67ovNxz-Y6o-1MULpd8u) (`ls_dq_cgp_exist_best_epoch124.ckpt`, Seed 2023)
* **LS-DQ-CGP（训练与推理未使用 existence head）**: `checkpoints/ls_dq_cgp_best_epoch139.ckpt` (Seed 2023)。该历史 checkpoint 的 state dict 保留了未训练的 `exist_head` 参数，但原实验以 `mr_only=True` 运行；本仓库按 checkpoint 训练协议识别它，而不按参数名误判。
* **DQ-CGP V3 released checkpoint (Intermediate Residual)**: `checkpoints/dq_cgp_v3_best_epoch86.ckpt` (Seed 2023)，对应 Test mAP **15.51**、AUROC **77.33**。

主表中的 DQ-CGPv3 使用另一轮从头复现得到的 Epoch 112 checkpoint，对应 Test mAP **17.72**、AUROC **76.23**；该 checkpoint 未随仓库发布。为避免跨 checkpoint 拼接，下表所有 DQ-CGPv3 指标均来自这一 reproduced checkpoint。已发布 Epoch 86 checkpoint 的完整指标见 [`results/test/metrics.json`](results/test/metrics.json)。

### Standard Test 官方 GMR 评测对比 (Seed 2023)

| 评估指标 | Matched retrained Baseline | Native Binding | DQ-CGPv3 reproduced | LS-DQ-CGP（未使用 Exist） | **LS-DQ-CGP + Exist** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Test mAP** | 6.14 | 14.46 | 17.72 | 16.65 | **18.07** |
| **mR@1** | 4.16 | 9.90 | 11.92 | 10.84 | **12.35** |
| **mR@3** | 6.48 | 16.13 | **19.35** | 18.08 | 18.74 |
| **mR@5** | 7.89 | 19.77 | 23.59 | 21.78 | **24.49** |
| **mR+@3（多时刻）** | 0.56 | 1.64 | 5.56 | 3.93 | **5.83** |
| **mR+@5（多时刻）** | 0.69 | 4.43 | **10.20** | 8.44 | 8.71 |
| **mIoU@1** | 12.30 | 25.42 | 28.80 | 25.97 | **30.03** |
| **mIoU@5** | 11.61 | 24.19 | 26.29 | 24.07 | **27.01** |
| **mIoU+@3（多时刻）** | 1.52 | 6.80 | 8.77 | 7.35 | **9.89** |
| **mIoU+@5（多时刻）** | 1.54 | 6.21 | 8.39 | 7.12 | **9.35** |
| **AUROC** | 71.87 | **76.69** | 76.23 | 75.04 | 75.83 |
| **G-mIoU@1** | 5.67 | 30.07 | 32.23 | 15.40 | **32.39** |
| **Best Val mAP** | 7.09 (ep11) | 19.55 (ep81) | **20.80 (ep112)** | 19.10 (ep139) | 19.99 (ep124) |

本次 LS-DQ-CGP + Exist 在 Test mAP 上达到 **18.07**，相对 DQ-CGPv3 reproduced 提高 0.35，相对 Native Binding 提高 3.61；同时取得最高的 mR@5、mIoU@1/5 和 mIoU+@3/5。DQ-CGPv3 reproduced 在 mR+@5 上领先，Native Binding 的 AUROC 最高。所有结果均为单 seed，0.35 mAP 的差距尚需多 seed 验证。

同 checkpoint 反事实评测中，Active / Static Bypass / Context Roll 的 mAP 分别为 **18.07 / 11.53 / 17.16**；三者 AUROC 均为 75.83，说明 existence 判别与语义排序消融已经解耦。

### 2.1 所有独立训练方法：相对 Baseline 的修改与 Val/Test 结果

以下各行都是独立训练后按 **Best Val MR-full-mAP** 选出的 checkpoint，再在相同的 1,036-query Standard Test 上评测。所有行均为 Seed 2023；DQ-CGP released 与 reproduced 是两次不同的训练。`GMR-CLS score` 明确标出归档 AUROC/G-mIoU 使用独立 `pred_exist_score` 还是检索 `window_score`。Test 列统一使用同一行 checkpoint 的指标。

| 方法 | 相对 Matched Baseline 的修改 | GMR-CLS score | Best Val mAP (epoch) | Test mAP | mR@5 | mR+@5 | G-mIoU@1 | AUROC | 证据 |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---|
| Matched Baseline | Moment-DETR-GMR 对照；无 DQ/LS 分支 | exist | 7.09 (11) | 6.14 | 7.89 | 0.69 | 5.67 | 71.87 | [factorial report](results/component_attribution_seed2023/RESULTS.md) |
| DQ-CGP V3 released | Query-specific temporal context + RCG/BPS/FRF + binding/route loss；在 D1-D2 间以 `beta=0.05` 注入 residual | exist | 19.02 (field 86) | 15.51 | 22.45 | 7.16 | 43.25 | 77.33 | [Val](results/val_best_metrics.json) / [Test](results/test/metrics.json) |
| DQ-CGP V3 reproduced | 与 released 同一方法，从头复现实验；checkpoint 不随仓库发布 | exist | **20.80 (112)** | 17.72 | 23.59 | **10.20** | 32.23 | 76.23 | [factorial report](results/component_attribution_seed2023/RESULTS.md) |
| Native Binding | 仅给原生 D1 cross-attention 加 matched binding loss；零新增推理参数 | exist | 19.55 (81) | 14.46 | 19.77 | 4.43 | 30.07 | 76.69 | [native report](native_binding_validation_lab/README.md) |
| LS-DQ-CGP | Native Binding + RCG/BPS/FRF late-semantic matcher，直接生成 D2 relevance logits；不使用独立 existence score | window | 19.10 (139) | 16.65 | 21.78 | 8.44 | 15.40 | 75.04 | [LS report](results/ls_dq_cgp_seed2023/RESULTS.md) |
| **LS-DQ-CGP + Exist** | LS-DQ-CGP + 独立 existence head，并用空 GT query 联合训练 | exist | 19.99 (124) | **18.07** | **24.49** | 8.71 | 32.39 | 75.83 | [LS+Exist report](results/ls_dq_cgp_exist_seed2023/RESULTS.md) |
| QAP | LS+Exist 中 BPS 的 6-token mean pooling 改为 query-conditioned attentive pooling | exist | 20.23 (135) | 14.49 | 19.70 | 4.79 | 30.58 | 74.98 | [artifacts](results/ls_dq_cgp_tap_exist_seed2023/) |
| Token-V1 | LS+Exist 增加由 $V_q$ 条件化的 pre-encoder text-token selector | exist | 19.83 (180) | 16.05 | 21.56 | 6.14 | 25.22 | 76.32 | [artifacts](results/token_ls_dq_cgp_exist_seed2023/) |
| Token-V2 | Token-V1 selector 的文本输入改为 multimodal encoder text memory | exist | **20.97 (138)** | 17.04 | 23.57 | 7.41 | 34.69 | 74.81 | [artifacts](results/token_ls_dq_cgp_v2_exist_seed2023/) |
| Encoder-Text LS | 移除 token selector；对 encoder text memory 做 masked mean 后送入 RCG/BPS/FRF | exist | 19.53 (100) | 16.38 | 24.17 | 7.94 | **36.10** | 76.12 | [artifacts](results/encoder_text_ls_dq_cgp_exist_seed2023/) |

Val 与 Test 的排序并不一致：QAP、Token-V2 和若干消融在 Val 上达到较高 mAP，但没有在 Test 上超过 LS-DQ-CGP + Exist。因此本仓库同时报告 Best Val 和固定 checkpoint 的 Test 指标，不以 Test 重新选择模型。

### 2.2 DQ-CGP 因子实验：Binding、Route 与 Injection

下面八组是完整的 `Binding (B) × Route (R) × Injection (I)` 从头训练实验。`B` 是 matched temporal binding loss，`R` 是 routing regularization，`I` 是 D1-D2 间 `beta=0.05` residual injection。`Native Binding` 不含 DQ 分支，作为第九个零新增参数对照。

| 变体 | B | R | I | 相对 Baseline 的实际修改 | Best Val mAP (epoch) | Test mAP | mR@5 | mR+@5 | G-mIoU@1 | AUROC |
|---|:---:|:---:|:---:|---|---:|---:|---:|---:|---:|---:|
| `baseline` | ✗ | ✗ | ✗ | 无 DQ 分支 | 7.09 (11) | 6.14 | 7.89 | 0.69 | 5.67 | 71.87 |
| `injection_only` | ✗ | ✗ | ✓ | 仅增加未受 binding/route 监督的 DQ residual | 8.13 (57) | 6.71 | 10.94 | 0.69 | 6.42 | 70.42 |
| `route_only` | ✗ | ✓ | ✗ | 训练 routing 分支与 route loss，但不注入 D2 | 8.30 (9) | 7.84 | 12.60 | 1.25 | 6.35 | 68.88 |
| `no_binding` | ✗ | ✓ | ✓ | Route + residual，无 matched binding loss | 8.51 (7) | 8.18 | 14.12 | 0.72 | 6.00 | 70.70 |
| `binding_only` | ✓ | ✗ | ✗ | DQ binding supervision，无 route loss、无 residual | 19.04 (159) | 14.91 | 20.67 | 4.45 | **35.03** | 76.08 |
| `no_route` | ✓ | ✗ | ✓ | Binding + residual，无 route loss | 20.02 (149) | 14.38 | 21.69 | 5.90 | 30.75 | **76.47** |
| `no_inject` | ✓ | ✓ | ✗ | Binding + route；训练 DQ 分支但 D2 接收原生 D1 state | 16.76 (77) | 12.94 | 17.33 | 6.81 | 27.21 | 74.09 |
| `full` | ✓ | ✓ | ✓ | 完整 DQ-CGP reproduced | **20.80 (112)** | **17.72** | **23.59** | **10.20** | 32.23 | 76.23 |
| `native_binding` | native | ✗ | ✗ | 原生 D1 attention + matched binding loss；无 DQ 参数 | 19.55 (81) | 14.46 | 19.77 | 4.43 | 30.07 | 76.69 |

这些单 seed 结果显示 Binding 是平均主效应最大的因素；Injection-only 仅比 baseline 高 0.57 mAP，而完整组合存在明显交互。完整效应计算与每轮日志见 [9 组因子归因报告](results/component_attribution_seed2023/RESULTS.md)。

### 2.3 LS-DQ-CGP 组件的从头重训练消融

这些变体均从头训练，保持 Seed、优化器、existence head、saliency supervision 和 Native Binding 配置一致，允许剩余模块重新适配。

| 方法 | 相对 Full LS+Exist 的修改 | Best Val mAP (epoch) | Test mAP | mR@5 | mR+@5 | G-mIoU@1 | AUROC |
|---|---|---:|---:|---:|---:|---:|---:|
| Full LS+Exist | 无修改 | 19.99 (124) | **18.07** | 24.49 | 8.71 | 32.39 | 75.83 |
| `bps_zero` | routed prompt $P_q$ 固定为 0；保留 FRF 的静态语义和局部视觉输入 | **21.17 (116)** | 17.37 | 23.51 | 6.59 | 26.46 | 76.51 |
| `rcg_uniform` | basis routing 权重固定为均匀分布 | 19.84 (92) | 16.77 | 24.76 | **9.14** | 25.50 | 75.42 |
| `frf_remove` | 移除学习式 FRF，仅保留 routed-prompt residual | 20.83 (131) | 16.74 | **25.16** | 8.72 | **38.19** | 75.44 |
| `bps_query_mean` | 每个 candidate prompt 替换为样本内 candidate 均值 | 19.63 (117) | 15.70 | 22.61 | 8.24 | 24.53 | 76.62 |
| `native_binding_exist_aligned` | 移除 late-semantic CGP/matcher，保留 D1 binding、existence 与 saliency | 18.55 (99) | 15.40 | 21.63 | 5.33 | 23.55 | **77.48** |
| Strict `Delta E_q=0` train | 全训练过程固定 semantic residual 为 0，保留 norm/matcher 架构 | 18.62 (72) | 14.89 | 20.12 | 8.52 | 24.25 | 77.40 |

Val 最优并不等于 Test 最优，例如 `bps_zero` 的 Val mAP 最高但 Test 仍低于 Full 0.70。完整定义、日志与预测见 [`ls_dq_cgp_ablation_lab/README.md`](ls_dq_cgp_ablation_lab/README.md) 和 [`strict_delta_zero_lab/README.md`](strict_delta_zero_lab/README.md)。

### 2.4 同 checkpoint 的推理反事实

下表不重新训练、不重新选择 checkpoint，只在已选最佳模型上改变一条计算路径；因此没有独立 Best Val。`Delta` 均相对同一 checkpoint 家族的 Active 模式。

| checkpoint（训练时 Best Val） | 推理模式 | 干预 | Test mAP | Delta | mR@5 | mR+@5 |
|---|---|---|---:|---:|---:|---:|
| LS no-Exist ep139 (19.10) | Active | 无 | 16.65 | 0.00 | 21.78 | 8.44 |
| LS no-Exist ep139 (19.10) | Static Bypass | 用全局静态文本替换 adapted semantics | 11.68 | -4.97 | 18.64 | 6.77 |
| LS+Exist ep124 (19.99) | Active | 无 | 18.07 | 0.00 | 24.49 | 8.71 |
| LS+Exist ep124 (19.99) | Static Bypass | 绕过 residual 与 `frf_norm` | 11.53 | -6.54 | 18.03 | 5.94 |
| LS+Exist ep124 (19.99) | Strict Delta-zero | 仅令 $\Delta E_q=0$，保留 `frf_norm`/matcher | 11.61 | -6.46 | 18.19 | 5.94 |
| LS+Exist ep124 (19.99) | Context Roll | 在 query 轴错配 $V_q$ | 17.16 | -0.91 | 23.87 | 7.51 |
| LS+Exist ep124 (19.99) | `rcg_uniform` | 强制均匀 routing | 17.39 | -0.68 | 23.39 | 8.61 |
| LS+Exist ep124 (19.99) | `bps_query_mean` | candidate prompt 取样本内均值 | 17.90 | -0.17 | 24.54 | 8.49 |
| LS+Exist ep124 (19.99) | `bps_zero` | prompt 置零 | 17.96 | -0.11 | 23.45 | 8.86 |
| LS+Exist ep124 (19.99) | `frf_remove` | 移除 FRF | 12.63 | -5.44 | 20.38 | 7.17 |
| QAP ep135 (20.23) | Active / UniformPrompt | attentive / uniform prompt pooling | 14.49 / 14.49 | 0.00 | 19.70 / 19.70 | 4.79 / 4.79 |
| Token-V1 ep180 (19.83) | Active / Token Static / Context Roll | selector active / 均匀 token / 错配 $V_q$ | 16.05 / 16.19 / 14.81 | 0.00 / +0.14 / -1.24 | 21.56 / 21.69 / 19.89 | 6.14 / 6.23 / 5.06 |
| Token-V2 ep138 (20.97) | Active / Uniform / Selector Roll | selector active / 均匀 token / 仅错配 selector context | 17.04 / 17.12 / 17.08 | 0.00 / +0.08 / +0.04 | 23.57 / 23.72 / 23.82 | 7.41 / 7.60 / 7.47 |
| Encoder-Text ep100 (19.53) | Active / PreEncoder / Context Roll | encoder mean / pre-encoder condition / 错配 $V_q$ | 16.38 / 16.49 / 14.55 | 0.00 / +0.11 / -1.83 | 24.17 / 23.90 / 21.66 | 7.94 / 7.72 / 5.03 |

推理反事实回答“这个已训练 checkpoint 是否依赖某条路径”；从头消融回答“去掉该路径后其余模块能补偿到什么程度”。两者不能混成同一种因果效应。

### 2.5 LS-DQ-CGP + QAP：Query-conditioned Attentive Prompt Pooling（探索性版本）

QAP 保留原有 `Bind → Adapt → Match` 路径，仅将 BPS 的 6-token mean pooling 替换为由 $[V_q;E_{static}]$ 条件化的 attention pooling。该实现采用 baseline-preserving initialization：$W_Q=0$、$b_Q=0$、$W_V=I$，因此训练初始时严格退化为原始 MeanPool；同时提供 `uniform_prompt_pool` 反事实，用于区分 attentive composition 与额外 Q/K/V 参数的贡献。该组合是 factorized composition：**Basis Routing × Prompt Position Attention**，不是对 $16\times6$ 个 basis tokens 独立路由。

当前 Seed 2023 结果如下。QAP 最佳验证 checkpoint 为 Epoch 135，训练在 Epoch 185 early stop。

| 方法 | Test mAP | mR@1 | mR@3 | mR@5 | mIoU@1 | G-mIoU@1 | AUROC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原 LS-DQ-CGP + Exist | **18.07** | **12.35** | **18.74** | **24.49** | **30.03** | **32.39** | **75.83** |
| QAP Active | 14.49 | 9.80 | 16.42 | 19.70 | 26.05 | 30.58 | 74.98 |
| QAP UniformPrompt | 14.49 | 9.80 | 16.42 | 19.70 | 26.05 | 30.58 | 74.98 |

QAP 的最佳验证 MR-full-mAP 为 **20.23**（原 LS-DQ-CGP 为 19.99），但该小幅验证集提升没有迁移到 Test；Active 与 UniformPrompt 的全部官方指标完全一致。因而当前实验不支持 attentive prompt composition 作为主方法，建议将其保留为负结果/消融。代码位于 `ls_dq_cgp_tap_lab/`，训练输出和 checkpoint 默认写入 `outputs/ls_dq_cgp_tap_exist_seed2023/`（不纳入 Git）。

QAP 的可审计训练与评估记录（不含 checkpoint 或预测 JSONL）已归档至 [`results/ls_dq_cgp_tap_exist_seed2023/`](results/ls_dq_cgp_tap_exist_seed2023/)：[`train.log`](results/ls_dq_cgp_tap_exist_seed2023/train.log)、[`val.log`](results/ls_dq_cgp_tap_exist_seed2023/val.log)、[`best_val_metrics.json`](results/ls_dq_cgp_tap_exist_seed2023/best_val_metrics.json)、[`Active metrics`](results/ls_dq_cgp_tap_exist_seed2023/test_active/metrics.json) 和 [`UniformPrompt metrics`](results/ls_dq_cgp_tap_exist_seed2023/test_uniform_prompt_pool/metrics.json)。

> Baseline 口径说明：主表的 6.14 是与因子实验统一配置重新训练的 matched baseline；仓库另保留了本地论文原始 Moment-DETR checkpoint 的结果（mAP 5.34、AUROC 70.25），见 [`results/baseline_test_metrics.json`](results/baseline_test_metrics.json)。两者来自不同训练 checkpoint，不应视为同一 baseline 的重复评测。

### 2.6 Token-V1：Pre-encoder Token-Selective LS-DQ-CGP（历史实现与日志）

Token-V1 实现 `Bind → Select → Adapt → Match`：每个 bound visual context $V_q$ 先对 pre-encoder 的有效文本 token 做 occurrence-conditioned selection，再进入 RCG/BPS/FRF。Seed 2023 的最佳 checkpoint 为 Epoch 180，最佳验证 MR-full-mAP 为 **19.83**。

| Test 模式 | mAP | mR@1 | mR@3 | mR@5 | mIoU@1 | G-mIoU@1 | AUROC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Active | 16.05 | 10.94 | 17.95 | 21.56 | 26.50 | 25.22 | 76.32 |
| Token Static | 16.19 | 10.95 | 17.92 | 21.69 | 26.74 | 25.30 | 76.32 |
| Context Roll | 14.81 | 9.84 | 15.64 | 19.89 | 23.66 | 24.76 | 76.32 |

训练、验证和完整测试运行日志已归档至 [`results/token_ls_dq_cgp_exist_seed2023/`](results/token_ls_dq_cgp_exist_seed2023/)，包括 [`train.log`](results/token_ls_dq_cgp_exist_seed2023/train.log)、[`val.log`](results/token_ls_dq_cgp_exist_seed2023/val.log)、[`run.log`](results/token_ls_dq_cgp_exist_seed2023/run.log) 及三种测试模式的指标文件。预测 JSONL 与 checkpoint 未上传。当前 `main` 分支中的 `ls_dq_cgp_token_lab/` 已升级为 Token-V2；若要核对 Token-V1 源码，应使用最后一个 V1 修复提交 [`946c505`](https://github.com/chinagalaxy2002/DQ-CGP/commit/946c505)，而不能直接用当前 V2 源码宣称复现 V1。

> 📌 **详细评测文件**：
> - [LS-DQ-CGP + Existence Head 完整结果与反事实分析](results/ls_dq_cgp_exist_seed2023/RESULTS.md)
> - [LS-DQ-CGP 结果与消融分析报告](results/ls_dq_cgp_seed2023/RESULTS.md)
> - [9组因子归因消融实验报告](results/component_attribution_seed2023/RESULTS.md)
> - [DQ-CGP V3 Test Metrics](results/test/metrics.json)
> - [DQ-CGP released-checkpoint beta sweep](results/dq_cgp_beta_sweep_seed2023/RESULTS.md)
> - [Decoder query-ownership diagnostic](results/query_ownership_seed2023/RESULTS.md)
> - [Causal occurrence-binding diagnostics](causal_occurrence_lab/results/RESULTS.md)

### 2.7 Token-V2：在同一 Multimodal Encoder Space 中选择文本

Token-V2 保留 `Bind → Select → Adapt → Match` 主路径、原始全局语义锚点 $E_{static}$、Native Binding Loss 和全部训练超参数，只把 selector 使用的文本表示从 pre-encoder projection 改为 multimodal encoder memory：

```text
Token-V1: V_q^(encoder) → TextTokens^(pre-encoder)
Token-V2: V_q^(encoder) → TextTokens^(encoder)
```

具体实现为 `txt_mem = memory[:, video_length:]`，并把 `txt_mem` 送入 token selector 和 local-semantic value aggregation；$E_{static}$ 仍由原始 `t_proj` 池化得到。Token-V2 还提供两个 inference-only counterfactual：

- `uniform_text_attention`：在有效文本 token 上强制均匀 attention，只移除 occurrence-specific token selection。
- `selector_context_roll`：只把滚动后的 $V_q$ 送给 selector，RCG/FRF 继续接收正确的 $V_q$。

最终训练在 Epoch 188 early stop，最佳 checkpoint 保持为 Epoch 138（checkpoint 字段为零基 `137`）；checkpoint 文件后续单独上传。该 checkpoint 的验证集 `MR-full-mAP` 为 **20.97**，`R1@0.5` 为 **34.90**，`R1@0.7` 为 **17.65**。同一 checkpoint 的 Standard Test 结果如下：

| Test 模式 | mAP | mR@1 | mR@3 | mR@5 | mIoU@1 | G-mIoU@1 | AUROC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Active | 17.04 | 11.29 | 18.65 | 23.57 | 27.56 | 34.69 | 74.81 |
| Uniform Text Attention | **17.12** | **11.41** | 18.73 | 23.72 | **27.87** | **34.75** | 74.81 |
| Selector-only Context Roll | 17.08 | 11.26 | **18.91** | **23.82** | 27.44 | 34.58 | 74.81 |

| Test 模式 | mR+@3 | mR+@5 | mIoU+@3 | mIoU+@5 |
| :--- | ---: | ---: | ---: | ---: |
| Active | 3.53 | 7.41 | 7.77 | 7.48 |
| Uniform Text Attention | **3.59** | **7.60** | **7.95** | 7.66 |
| Selector-only Context Roll | 3.56 | 7.47 | 7.92 | **7.76** |

Active 没有优于 Uniform Text Attention，Selector-only Context Roll 也没有造成稳定退化。因此，这个 checkpoint 不支持“正确 occurrence context 通过 token selector 改善 language selection”的因果结论。Token-V2 修复了表示阶段不一致，但当前 selector 仍应视为探索性负结果。Epoch 138 之后直到 Epoch 188 均未产生新的 validation-best checkpoint。

Epoch 138 训练/验证快照和测试记录已归档至 [`results/token_ls_dq_cgp_v2_exist_seed2023/`](results/token_ls_dq_cgp_v2_exist_seed2023/)：包括 [`train.log`](results/token_ls_dq_cgp_v2_exist_seed2023/train.log)、[`val.log`](results/token_ls_dq_cgp_v2_exist_seed2023/val.log)、Epoch 138 验证指标，以及 Active、Uniform Text Attention 和 Selector-only Context Roll 的 `metrics.json`/`result.json`。完整训练日志已核对到 Epoch 188；归档日志仍保留产生最佳 checkpoint 时的固定快照。

### 2.8 Encoder-Text LS-DQ-CGP

Encoder-Text LS 是 Token-V2 的单一减法实验：移除 occurrence-conditioned token selector，直接对 multimodal encoder 的有效文本 memory 做 masked mean，得到 $E_{enc}$，并使用

```text
V_q + E_enc → RCG → BPS (原 MeanPool) → FRF
E_static + semantic_delta → E_adapt
```

其中 $E_{static}$ 继续作为 pre-encoder global semantic anchor。训练保持 Seed 2023、`lr=5e-5`、`native_bind_coef=0.2`、16 个 basis、prompt length 6 和 existence head 开启；代码位于 `ls_dq_cgp_encoder_text_lab/`。

最终训练在 Epoch 150 early stop，最佳验证 checkpoint 为 Epoch 100，Val MR-full-mAP 为 **19.53**。同一 checkpoint 的 Standard Test 结果如下：

| Test 模式 | mAP | mR@1 | mR@3 | mR@5 | mIoU@1 | mIoU@3 | mIoU@5 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Active | 16.38 | 9.54 | 18.84 | 24.17 | 26.07 | 23.56 | 23.46 |
| PreEncoderCondition | **16.49** | 9.59 | **19.48** | 23.90 | **26.10** | **23.87** | **23.70** |
| ContextRoll | 14.55 | 8.10 | 15.33 | 21.66 | 21.59 | 19.57 | 19.48 |

`PreEncoderCondition` 略高于 Active，因此本实验没有证明 $E_{enc}$ 相比 $E_{static}$ 在 Test 上具有额外的正因果贡献；但 ContextRoll 从 16.38 降至 14.55，继续支持正确 occurrence-specific $V_q$ 对后续语义适配的重要性。

训练与评测日志及指标文件已归档至 [`results/encoder_text_ls_dq_cgp_exist_seed2023/`](results/encoder_text_ls_dq_cgp_exist_seed2023/)，包括 [`train.log`](results/encoder_text_ls_dq_cgp_exist_seed2023/train.log)、[`val.log`](results/encoder_text_ls_dq_cgp_exist_seed2023/val.log)、Active、PreEncoderCondition 和 ContextRoll 的指标与运行元数据。checkpoint 和预测 JSONL 不上传。

---

### 2.9 组件级消融实验（LS-DQ-CGP Ablation Lab）

组件级消融代码集中在 [`ls_dq_cgp_ablation_lab/`](ls_dq_cgp_ablation_lab/)，不修改生产模型文件，提供统一 Seed 2023、优化器、existence head、saliency supervision 和 Native Binding 配置下的可复现实验：

* `rcg_uniform`：RCG basis 权重固定为均匀路由；
* `bps_query_mean` / `bps_zero`：分别消除 candidate-specific prompt 差异或将 prompt 置零；
* `frf_remove`：移除学习式 FRF 融合，仅保留 routed prompt residual；
* `native_binding_exist_aligned`：保留 D1 Native Binding 与 existence head，移除 late-semantic CGP 和 semantic matcher；
* `delta-zero`：严格固定 $\Delta E_q=0$ 的独立实验见 `strict_delta_zero_lab/`。

每个变体均提供训练、测试和 tmux 启动脚本；详细干预定义、推理级 sanity check、最终重训 Test 结果和运行命令见 [`ls_dq_cgp_ablation_lab/README.md`](ls_dq_cgp_ablation_lab/README.md)。五组重训的指标、预测 JSONL 与日志已归档在 `outputs/ls_ablation_*` 和 `logs/`；checkpoint 不纳入 Git。

### 2.10 DQ-CGP 因果诊断与负结果

为避免只报告正向结果，本仓库同时归档以下同 checkpoint 反事实与诊断实验：

| 诊断 | checkpoint | 主要观察 | 结论边界 |
|---|---|---|---|
| [Test-time beta sweep](results/dq_cgp_beta_sweep_seed2023/RESULTS.md) | released Epoch 86 | `beta=0` mAP 15.85，高于训练值 `beta=0.05` 的 15.51；更大 beta 整体降低检索 mAP | 当前 released checkpoint 的 mAP 不支持 residual injection 带来正贡献；部分 GMR gate 指标并非同方向 |
| [Query ownership](results/query_ownership_seed2023/RESULTS.md) | reproduced Epoch 112 | Baseline / DQ active / DQ beta-zero retention 为 73.97% / 89.04% / 89.04%；active 与 beta-zero 仅两个 qid 的 D2 assignment 不同 | DQ 与独立 baseline 的差异不能单独归因于 beta adapter |
| [Causal occurrence binding](causal_occurrence_lab/results/RESULTS.md) | existing checkpoints | DQ active 与 beta-zero 的分类概率、span 和 Top-5 排名变化较小；stripped 与 beta-zero 数值等价 | 这是 raw-span checkpoint diagnostic，不是完整多 seed 因果训练结果 |

这些结果与 LS-DQ-CGP 的主结果不冲突：它们说明原版 DQ-CGP 的小幅中间层 residual 在现有 checkpoint 上因果效应有限，也解释了为什么后续工作转向直接主导语义排序的 late-semantic 设计。所有比较均保留 checkpoint 身份和评估口径，不将跨 checkpoint 差异写成模块级因果结论。

`causal_occurrence_lab` 还实现了 `no_bind`、`supervision_only`、`union_bind`、`wrong_bind`、`no_route`、`architecture_only` 和 `native_bind` 等训练控制，但当前 GitHub 仅包含 existing-checkpoint 分析和 one-batch smoke artifacts，没有可作为正式结果的完整多 seed Val/Test 训练。因此这些名称不进入上面的独立训练结果表，也不宣称性能结论。

---

## 3. 环境与 Soccer-GMR 特征

Soccer-GMR 采用 gated access、NDA 和禁止再分发条款，因此本仓库不包含视频或预计算特征。请从 Soccer-GMR 官方 gated release 获取 `feature.tar`，并整理为：

```text
Soccergmr/
├── clip/
│   └── <video_id>.npz          # key: features
├── slowfast/
│   └── <video_id>.npz          # key: features
└── clip_text/
    └── qid<query_id>.npz       # keys: last_hidden_state, attention_mask
```

Standard split 标签已包含在 `data/label/Standard/{train,val,test}.jsonl`。训练前应确认 `Soccergmr/clip`、`Soccergmr/slowfast` 与 `Soccergmr/clip_text` 三个目录存在。

```bash
conda create -n dq-cgp python=3.9 -y
conda activate dq-cgp

# 安装依赖
pip install -r requirements.txt
```

---

## 4. 快速复现与训练

### 4.1 一键复现 LS-DQ-CGP + Existence Head（推荐）

下面的脚本依次执行训练、Active、Static Bypass 和 Context Roll 测试。可通过 `PYTHON` 指定解释器，第一个位置参数指定 GPU：

```bash
PYTHON=/path/to/python bash ls_dq_cgp_lab/run_experiment_with_exist.sh 0
```

等价的训练命令为：

```bash
python ls_dq_cgp_lab/train_ls_dq_cgp.py \
  --output outputs/ls_dq_cgp_exist_seed2023 \
  --gpu 0 \
  --seed 2023 \
  --epochs 400 \
  --lr 5e-5 \
  --native_bind_coef 0.2 \
  --use_exist_head
```

### 4.2 单独评测与反事实消融
```bash
# 1. 运行 Active 模式测试
python ls_dq_cgp_lab/evaluate_ls_dq_cgp.py \
  --checkpoint /path/to/ls_dq_cgp_exist_best_epoch124.ckpt \
  --output outputs/ls_dq_cgp_exist_test_active \
  --split test \
  --gpu 0

# 2. 运行 Static Bypass 模式反事实测试
python ls_dq_cgp_lab/evaluate_ls_dq_cgp.py \
  --checkpoint /path/to/ls_dq_cgp_exist_best_epoch124.ckpt \
  --output outputs/ls_dq_cgp_exist_test_bypass \
  --split test \
  --static_bypass \
  --gpu 0

# 3. 错配 Query 局部视觉上下文
python ls_dq_cgp_lab/evaluate_ls_dq_cgp.py \
  --checkpoint /path/to/ls_dq_cgp_exist_best_epoch124.ckpt \
  --output outputs/ls_dq_cgp_exist_test_context_roll \
  --split test \
  --context_roll \
  --gpu 0
```

### 4.3 QAP 探索性版本与 UniformPrompt 反事实

```bash
# 训练 QAP + Existence Head（baseline-preserving initialization）
python ls_dq_cgp_tap_lab/train_ls_dq_cgp.py \
  --output outputs/ls_dq_cgp_tap_exist_seed2023 \
  --gpu 0 \
  --seed 2023 \
  --epochs 400 \
  --lr 5e-5 \
  --native_bind_coef 0.2 \
  --use_exist_head

# Active QAP
python ls_dq_cgp_tap_lab/evaluate_ls_dq_cgp.py \
  --checkpoint outputs/ls_dq_cgp_tap_exist_seed2023/best.ckpt \
  --output outputs/ls_dq_cgp_tap_exist_seed2023/test_active \
  --split test \
  --gpu 0

# UniformPrompt：仅强制 alpha_{q,p}=1/6，其余模块不变
python ls_dq_cgp_tap_lab/evaluate_ls_dq_cgp.py \
  --checkpoint outputs/ls_dq_cgp_tap_exist_seed2023/best.ckpt \
  --output outputs/ls_dq_cgp_tap_exist_seed2023/test_uniform_prompt_pool \
  --split test \
  --uniform_prompt_pool \
  --gpu 0
```

### 4.4 Token-V2 训练与两个 selector 反事实

下面的脚本使用与 Token-V1 相同的 loss 和超参数，依次运行 Token-V2 训练、Active、Uniform Text Attention 和 Selector-only Context Roll：

```bash
PYTHON=/path/to/python \
OUTPUT=outputs/token_ls_dq_cgp_v2_exist_seed2023 \
bash ls_dq_cgp_token_lab/run_experiment_with_exist.sh 0
```

也可以使用同一 checkpoint 单独执行反事实评估：

```bash
# Active
python ls_dq_cgp_token_lab/evaluate_ls_dq_cgp.py \
  --checkpoint /path/to/ls_dq_cgp_token_v2_exist_best_epoch138.ckpt \
  --output outputs/token_v2/test_active \
  --split test --gpu 0

# Uniform Text Attention
python ls_dq_cgp_token_lab/evaluate_ls_dq_cgp.py \
  --checkpoint /path/to/ls_dq_cgp_token_v2_exist_best_epoch138.ckpt \
  --output outputs/token_v2/test_uniform_text_attention \
  --split test --uniform_text_attention --gpu 0

# Selector-only Context Roll
python ls_dq_cgp_token_lab/evaluate_ls_dq_cgp.py \
  --checkpoint /path/to/ls_dq_cgp_token_v2_exist_best_epoch138.ckpt \
  --output outputs/token_v2/test_selector_context_roll \
  --split test --selector_context_roll --gpu 0
```

---

## 5. 代码结构

```text
ls_dq_cgp_lab/                           # LS-DQ-CGP 独立实验与核心代码
├── cgp_module.py                        # Late-Semantic CGP 核心模块 (RCG, BPS, FRF, Matcher)
├── ls_dq_cgp_model.py                   # 模型组装与 Native Binding Loss 注入
├── train_ls_dq_cgp.py                   # 训练入口
├── evaluate_ls_dq_cgp.py                 # 官方 GMR 指标评测与反事实评估
├── run_experiment.sh                    # 一键训练与评估流水线
├── run_experiment_with_exist.sh         # Exist 版本：训练及三种 Test 模式
└── test_ls_dq_cgp.py                    # 单元回归测试

ls_dq_cgp_tap_lab/                       # QAP 探索性版本（含 UniformPrompt 反事实）
├── cgp_module.py                        # RCG, BPS, QAP, FRF 与语义匹配
├── ls_dq_cgp_model.py                   # 模型组装、Native Binding Loss、QAP diagnostic
├── train_ls_dq_cgp.py                   # QAP 训练入口
├── evaluate_ls_dq_cgp.py                # Active / UniformPrompt / 其他反事实评估
└── test_ls_dq_cgp.py                    # QAP 形状、初始化、梯度回归测试

ls_dq_cgp_token_lab/                     # Token-Selective LS-DQ-CGP
├── cgp_module.py                        # Token-V2 selection、Uniform Text 与 selector-only roll
├── ls_dq_cgp_model.py                   # Encoder text memory、Native Binding Loss、模型组装
├── train_ls_dq_cgp.py                   # Token-Selective 训练入口
├── evaluate_ls_dq_cgp.py                # Active / Uniform Text / Selector-only Roll 评估
├── run_experiment_with_exist.sh         # Token-V2 Exist 训练及三种 Test 模式
└── test_ls_dq_cgp.py                    # Token selection 回归测试

experiments/vmr_cgp/                     # 原版 DQ-CGP V3 代码
models/moment_detr_gmr/                  # Moment-DETR-GMR 主干实现
training/moment_detr_gmr/                # 数据集与基础训练模块
results/ls_dq_cgp_seed2023/              # LS-DQ-CGP 评测日志与指标记录
results/ls_dq_cgp_exist_seed2023/        # Exist 版本完整日志、预测与评测报告
results/token_ls_dq_cgp_exist_seed2023/  # Token-Selective 训练与测试日志、指标
results/token_ls_dq_cgp_v2_exist_seed2023/ # Token-V2 最佳点快照与测试指标
results/dq_cgp_beta_sweep_seed2023/       # released checkpoint 的 beta 反事实
results/query_ownership_seed2023/         # reproduced checkpoint 的 query ownership 诊断
causal_occurrence_lab/results/            # occurrence-binding 因果诊断与 smoke artifacts
ls_dq_cgp_ablation_lab/                  # RCG/BPS/FRF 与严格对齐 Native Binding+Exist 消融
checkpoints/                             # 发布 Checkpoint 与 SHA256SUMS
```

---

## Citation

```bibtex
@article{ding2026retrieving,
  title={Retrieving Any Relevant Moments: Benchmark and Models for Generalized Moment Retrieval},
  author={Ding, Yiming and Cao, Siyu and Jiao, Luyuan and Li, Yixuan and Wang, Zitong and Liu, Zhiyong and Zhang, Lu},
  journal={arXiv preprint arXiv:2605.02623},
  year={2026}
}
```

## License
MIT License.
