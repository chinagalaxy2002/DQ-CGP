# LS-DQ-CGP 分桶诊断评测方案

## 1. 目标

本文档为完整 LS-DQ-CGP 模型与受控消融模型设计细粒度测试方案，目标不只是比较总体分数，而是回答以下问题：

1. LS-DQ-CGP 擅长哪些类型的目标，例如极短、中等、较长、单目标或多目标？
2. RCG、BPS、FRF 和完整 late-semantic prediction path 分别在哪些样本上产生贡献？
3. 性能提升来自候选定位与排序，还是来自存在性判断及阈值校准？
4. 模型是否利用了数据源、查询模板或正负样本分布等非预期捷径？

评测采用“能力维度 × 难度分桶 × 成对消融”的组织方式。所有 Test 分桶边界必须预先由 Train/Val 数据确定，不能根据 Test 表现反复调整。

## 2. 被评估的模型与协议

### 2.1 完整模型和重训练消融

| 模型 | 干预 | 主要问题 |
|---|---|---|
| `full` | 无 | 完整 LS-DQ-CGP 的性能与能力边界是什么？ |
| `rcg_uniform` | 将候选特定路由替换为均匀 basis 权重 | 学习得到的 candidate-specific routing 是否有效？ |
| `bps_query_mean` | 同一样本内所有 DETR candidates 共用平均 prompt | 候选之间的 BPS 差异是否有效？ |
| `bps_zero` | 将 routed prompt 置零 | BPS prompt 是否在 FRF 直接输入之外提供信息？ |
| `frf_remove` | 删除 FRF MLP，保留 routed prompt residual | 学习得到的多模态融合是否必要？ |
| `native_binding_exist_aligned` | 删除完整 late-semantic CGP 和 semantic matcher | 完整 late-semantic prediction path 的贡献是什么？ |

重训练模型用于估计“删除模块以后，剩余模块充分补偿时的可实现性能”。主要成对差值为：

- `full - native_binding_exist_aligned`：完整 late-semantic 路径贡献；
- `full - rcg_uniform`：candidate-specific RCG 贡献；
- `full - bps_query_mean`：candidate-specific BPS 贡献；
- `full - bps_zero`：routed prompt 的总体贡献；
- `full - frf_remove`：FRF 贡献。

### 2.2 Inference-only 干预

在同一个训练完成的 Full checkpoint 上执行 `rcg_uniform`、`bps_query_mean`、`bps_zero` 和 `frf_remove`，用于测量该 checkpoint 对各模块的实际依赖程度。

Inference-only 和重训练消融回答不同的因果问题，必须分别报告，不能将二者混合平均：

- Inference-only：已训练 Full 模型是否依赖该组件；
- Retrained：删除该组件后，其他模块能否学习补偿。

### 2.3 反事实检查

以下结果作为机制有效性的辅助证据，不与组件消融合并排名：

- `static_bypass`：使用静态文本语义替换动态适配语义；
- `context_roll`：将 candidate 的局部视觉上下文循环错配给其他 candidate。

## 3. 数据概况与关键混杂因素

Soccer-GMR Standard Test 共 1,036 条查询：

- 正例 544 条，负例 492 条；
- 单目标 384 条；
- 双目标 128 条；
- 三目标及以上 32 条；
- SportsMoments 216 条，全部为正例且全部为单目标；
- WorldCup 820 条，包括 328 条正例和全部 492 条负例。

### 3.1 不能直接进行全局长短目标比较

SportsMoments 的目标主要为 2--8 秒可变长度单目标，而 WorldCup 正例大多被扩展为约 8 秒窗口，同时包含全部多目标查询。因此，全局目标长度同时混入了：

- 数据源；
- 查询模板；
- 正负样本比例；
- 目标数量；
- 动作类别和标注方式。

正式的长短目标结论应在 SportsMoments 内部得出。WorldCup 的约 8 秒窗口应单独报告，不能将跨数据源差异解释为纯长度能力。

### 3.2 全局存在性指标可能受到数据源捷径影响

SportsMoments Test 全部为正例，全部负例均来自 WorldCup。模型可能利用简单查询模板或数据源差异判断正负，导致全局 AUROC 高估真实存在性能力。因此：

- 主要存在性结果使用 WorldCup 内部 AUROC/AUPRC；
- 同时计算 major action 内部的 AUROC/AUPRC；
- 全局 AUROC 仅作为与既有结果对齐的补充指标。

## 4. Seed-2023 分桶结果

以下结果由仓库中现有 Test prediction JSONL 离线复评得到。本方案固定使用 Seed 2023，不要求额外训练其他随机种子；同一 Test qid 上的 paired bootstrap 用于估计分桶差值的不确定性。

### 4.1 完整路径在不同目标长度上的表现

为排除数据源混杂，长度分析仅使用 SportsMoments 正例。

| SportsMoments 目标长度 | 样本数 | Full mAP | Native mAP | `Full - Native` |
|---|---:|---:|---:|---:|
| ≤3 秒 | 81 | 3.71 | 7.28 | **-3.57** |
| 4--5 秒 | 72 | 26.30 | 22.07 | +4.23 |
| 6--8 秒 | 37 | 36.40 | 20.17 | **+16.23** |
| >8 秒 | 26 | 26.55 | 17.74 | +8.81 |

初步假设：Full LS-DQ-CGP 的优势主要出现在中等和较长目标，而不是极短目标。6--8 秒和 >8 秒桶样本较少，正式主表应合并为 `≥6 秒`，细分结果只作探索性分析。

### 4.2 目标数量、位置和边界

| 分桶 | 样本数 | Full mAP | Native mAP | `Full - Native` |
|---|---:|---:|---:|---:|
| 单目标 | 384 | 18.27 | 14.88 | +3.39 |
| 双目标 | 128 | 16.53 | 16.06 | +0.47 |
| 三目标及以上 | 32 | 21.75 | 19.02 | +2.73 |
| Early | 96 | 18.32 | 14.05 | +4.27 |
| Middle | 357 | 18.53 | 16.18 | +2.35 |
| Late | 91 | 15.98 | 13.77 | +2.21 |
| 距视频边界 ≤10% | 112 | 15.49 | 14.71 | +0.78 |
| 视频内部 | 432 | 18.73 | 15.58 | +3.15 |

初步假设：当前 Full 模型在单目标和视频内部目标上的增益更稳定；双目标和边界附近目标可能仍是主要能力缺口。三目标及以上只有 32 条，应标记为探索性结果。

### 4.3 数据源和动作类别

| 分桶 | 正例数 | Full mAP | Native mAP | `Full - Native` |
|---|---:|---:|---:|---:|
| SportsMoments | 216 | 19.59 | 15.68 | +3.91 |
| WorldCup | 328 | 17.06 | 15.21 | +1.85 |
| header | 105 | 21.92 | 14.91 | +7.01 |
| clearance | 70 | 25.88 | 18.43 | +7.45 |
| substitution | 39 | 18.17 | 10.64 | +7.53 |
| block | 66 | 14.22 | 15.21 | -0.99 |
| dribble | 42 | 10.37 | 11.80 | -1.43 |
| shot | 39 | 20.92 | 22.35 | -1.43 |

初步假设：late-semantic 路径可能更适合 header、clearance 和 substitution，而 block、dribble 等细粒度、持续时间短或视觉差异较小的动作仍然困难。样本数小于 50 的动作只作为探索性结果。

### 4.4 总体定位和存在性结果

| 模型 | Test mAP | AUROC |
|---|---:|---:|
| Full LS-DQ-CGP | 18.07 | 75.83 |
| `native_binding_exist_aligned` | 15.40 | 77.48 |

这里的 `native_binding_exist_aligned` 是本轮 LS-DQ-CGP 受控消融中的对齐重训练模型，与仓库早期总体表中的 Native Binding checkpoint 不是同一次训练。Full 的总体提升主要来自正例定位/排序，不是存在性分类。进一步在 WorldCup 内部计算时，Full AUROC 为 59.92，`native_binding_exist_aligned` 为 62.64，说明全局 AUROC 很可能受到数据源或查询模板捷径影响。

### 4.5 与主要 baseline 的总体对比

仓库已有的 Seed-2023 Standard Test 结果可用于总体 baseline 对比。历史总体表使用 reproduced DQ-CGPv3 checkpoint；本次对发布 checkpoint 的重测结果单独列出，避免跨 checkpoint 拼接。

| 方法 | mAP | mR@1 | mR@3 | mR@5 | mR+@5 | mIoU@1 | mIoU@5 | mIoU+@5 | AUROC | G-mIoU@1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Moment-DETR-GMR Baseline | 6.14 | 4.16 | 6.48 | 7.89 | 0.69 | 12.30 | 11.61 | 1.54 | 71.87 | 5.67 |
| Native Binding | 14.46 | 9.90 | 16.13 | 19.77 | 4.43 | 25.42 | 24.19 | 6.21 | **76.69** | 30.07 |
| DQ-CGPv3（reproduced checkpoint，历史总体结果） | 17.72 | 11.92 | **19.35** | 23.59 | **10.20** | 28.80 | 26.29 | 8.39 | 76.23 | 32.23 |
| DQ-CGPv3（发布 checkpoint，Test 重测） | 15.51 | 9.65 | 16.91 | 22.45 | 7.16 | 25.84 | 23.56 | 6.93 | **77.33** | **43.25** |
| LS-DQ-CGP（无 Exist） | 16.65 | 10.84 | 18.08 | 21.78 | 8.44 | 25.97 | 24.07 | 7.12 | 75.04 | 15.40 |
| **LS-DQ-CGP + Exist** | **18.07** | **12.35** | 18.74 | **24.49** | 8.71 | **30.03** | **27.01** | **9.35** | 75.83 | **32.39** |

相对 Moment-DETR-GMR Baseline，LS-DQ-CGP + Exist 的 mAP 提高 11.93，mR@5 提高 16.60，mIoU@1 提高 17.73，G-mIoU@1 提高 26.72。相对 Native Binding，mAP 提高 3.61；相对 reproduced DQ-CGPv3，mAP 提高 0.35。DQ-CGPv3 的 reproduced checkpoint 仍在 mR@3 和多目标 mR+@5 上更强；发布 checkpoint 的重测则有更高 AUROC 与 G-mIoU@1，但更低 mAP。两套 DQ-CGPv3 checkpoint 不可混合比较。当前结果支持 LS-DQ-CGP 在总体定位质量、Top-5 检索和多目标 IoU 方面更好，但不能声称它在多目标召回或存在性判断上全面领先。

正式分桶对比建议至少纳入以下三层 baseline：

1. Moment-DETR-GMR Baseline：衡量完整方法相对原始主干的总收益；
2. Native Binding：控制 D1 binding supervision 后，衡量 late-semantic inference path 的增量；
3. DQ-CGPv3：比较 late-semantic 与 intermediate residual 两种 CGP 注入位置的能力边界。

Plain Moment-DETR-GMR 与早期 Native Binding 仅保留了归档的总体 Test 指标，不能据此推测其分桶表现。DQ-CGPv3 的发布 checkpoint 已在 `outputs/dq_cgp_v3_seed2023/test_recheck/` 重测并导出逐 Test-query prediction JSONL，因此可用于 Full 与 DQ-CGPv3（发布 checkpoint）的分桶比较；该结果不与 reproduced DQ-CGPv3 checkpoint 的总体表混用。

## 5. 正式分桶定义

### 5.1 主分桶

| 维度 | 分桶定义 | Test 数量 | 主要研究问题 |
|---|---|---:|---|
| 目标长度 | SportsMoments：≤3s / 4--5s / ≥6s | 81 / 72 / 63 | 极短、中等和较长目标能力 |
| 目标数量 | 1 / 2 / ≥3 | 384 / 128 / 32 | 单目标与多目标检索能力 |
| 时间位置 | normalized center：Early / Middle / Late | 96 / 357 / 91 | 是否存在位置偏置 |
| 边界难度 | 最近 GT 距视频边界 ≤10% / >10% | 112 / 432 | 边界截断和位置编码能力 |
| 多目标最近间距 | ≤10% / 10%--30% / >30% 视频长度 | 80 / 40 / 40 | 密集与分散目标检索 |
| 多目标总体跨度 | ≤1/3 / 1/3--2/3 / >2/3 视频长度 | 84 / 52 / 24 | 跨长时间范围搜索能力 |
| 数据域 | SportsMoments / WorldCup | 216 / 328 正例 | 域和标注形式差异 |
| 动作类型 | 主要 action type 分别统计 | 每类约 39--105 正例 | 语义类别优势与缺口 |
| 查询长度 | WorldCup 内：7 / 8 / ≥9 词 | 129 / 382 / 309 总样本 | 排除数据源后的语言复杂度 |
| 存在性 | WorldCup 总体及 major action 内正/负 | 820 总样本 | 排除数据源捷径后的拒识能力 |

### 5.2 探索性分桶

- SportsMoments 的 `6--8 秒` 与 `>8 秒`；
- 三目标、四目标和五目标分别统计；
- 动作类别中 Test 正例数小于 50 的类别；
- 多目标总体跨度大于 2/3 视频长度；
- 查询措辞模板、队伍实体、稀有动作和失败案例类别。

探索性分桶用于生成假设。其结论必须同时给出样本量和 paired bootstrap 区间，避免将小样本波动解释为稳定能力差异。

## 6. 每类分桶使用的指标

### 6.1 正例定位桶

主要指标：

- mAP，作为候选定位与排序的 headline metric；
- mR@1、mR@3、mR@5；
- mIoU@1、mIoU@3、mIoU@5。

多目标专用指标：

- mR+@3、mR+@5；
- mIoU+@3、mIoU+@5；
- 独立 GT 命中数量；
- 候选窗口重复率和平均两两 IoU。

建议新增诊断指标：

- Oracle Recall@10：忽略候选分数，只判断生成的 10 个窗口是否覆盖 GT；
- Ranking Gap：Oracle Recall@10 与实际 Recall@K 的差值；
- Boundary Error：预测 start/end 与匹配 GT start/end 的平均绝对误差；
- Duration Bias：预测窗口时长与 GT 时长之比。

Oracle 能力较高而实际 Recall 较低表示主要问题在 semantic ranking；两者都低表示 span proposal 本身不足。

### 6.2 含正负样本的存在性桶

阈值无关指标：

- AUROC；
- AUPRC；
- Brier Score；
- Expected Calibration Error（ECE）。

阈值相关指标：

- Rej-F1；
- Accuracy；
- G-mIoU@1、G-mIoU@3、G-mIoU@5。

每个模型的存在性阈值只能在 Val 上选择，然后冻结到 Test。应同时给出固定公共阈值结果和 Val 校准阈值结果，以判断性能差异来自模型能力还是校准差异。

## 7. 统计协议

### 7.1 固定单种子协议

所有正式比较固定使用 Seed 2023，并保持以下条件一致：

- 相同的 Standard train/val/test split；
- 相同的输入特征、优化器、学习率、batch size 和 early-stopping 规则；
- 相同的 checkpoint 选择指标；
- 相同的后处理、最大候选数和 IoU thresholds；
- 存在性阈值只在 Val 上确定，再冻结到 Test。

该协议估计的是 Seed-2023 条件下的模型与组件差异，不估计训练随机性带来的跨种子方差。文档和论文中应明确写明所有结果均为单种子。

### 7.2 配对置信区间

对 Full 与每个消融模型按相同 qid 进行 paired bootstrap：

- bootstrap 次数：10,000；
- 报告 `Full - Ablation` 的均值和 95% CI；
- Full、baseline 和消融必须在同一组 qid 上配对重采样；
- 多桶显著性检验采用 Holm correction。

### 7.3 样本量规则

- `n ≥ 50`：允许作为主结论；
- `20 ≤ n < 50`：标记为 exploratory；
- `n < 20`：合并分桶或只报告案例，不进行独立能力宣称。

每张表必须同时报告绝对分数、样本量、相对 Full/Native 的差值和置信区间。

## 8. 模块机制诊断

仅有分桶性能可以说明“在哪里有效”，还需要内部统计解释“为什么有效”。建议在推理时按 qid 保存：

| 内部量 | 统计方式 | 对应模块与解释 |
|---|---|---|
| `basis_weights` 路由熵 | 每个 candidate 的 entropy 和 effective basis count | RCG 是否进行候选特定选择 |
| Candidate prompt 距离 | 同一查询内 pooled prompt 的平均 pairwise cosine distance | BPS 是否形成候选多样性 |
| FRF 残差比例 | `||E_adapt-E_static|| / ||E_static||` | FRF 在哪些桶中进行更强适配 |
| Semantic score margin | 匹配/未匹配 candidate 的分数差 | matcher 是否改进候选排序 |
| D1 GT attention mass | attention 落在 GT 窗口内的总质量 | 局部视觉绑定是否准确 |
| Candidate span diversity | 窗口两两 IoU、独立 GT 命中数 | 多目标检索是否发生候选塌缩 |

重点检验以下假设：

1. RCG 在目标跨度大、候选异质性高的查询上贡献更大；
2. `bps_query_mean` 会降低 candidate prompt 和预测窗口的多样性；
3. FRF 在需要强视觉-语义适配的动作及中长目标上贡献更大；
4. 极短目标的主要瓶颈可能来自 span resolution，而不是 semantic routing；
5. Full 的存在性能力没有随定位能力同步提升。

## 9. 推荐结果表与图

### 9.1 必做表格

1. Overall：Moment-DETR-GMR、Native Binding、DQ-CGPv3、Full LS-DQ-CGP 和所有消融的 mAP、mR、mIoU、mR+、mIoU+、AUROC 和 G-mIoU；
2. Length：SportsMoments 内长度主桶；
3. Multiplicity：目标数量、最近间距和总体跨度；
4. Position：Early/Middle/Late 与 boundary/interior；
5. Semantic：数据源和 major action；
6. Existence：WorldCup 总体及 action-conditioned AUROC/AUPRC；
7. Inference-only 与 retrained 两张独立消融表。

### 9.2 必做图形

1. `模型/消融 × 分桶` 的 ΔmAP heatmap；
2. SportsMoments 目标长度—性能折线图，带 bootstrap CI；
3. 多目标间距—mR+/mIoU+ 曲线；
4. WorldCup 存在性 reliability diagram；
5. 模块内部量与 Full 增益的相关性图。

## 10. 建议的离线分析程序

建议新增统一入口：

```text
ls_dq_cgp_ablation_lab/evaluate_buckets.py
```

输入：

- Train/Val/Test GT JSONL；
- 一个或多个命名 prediction JSONL；
- Val 上确定的分桶定义和存在性阈值；
- bootstrap 次数与随机种子。

建议输出：

```text
bucket_definition.json
bucket_metrics.csv
paired_deltas.csv
bootstrap_confidence_intervals.csv
threshold_calibration.json
figures/delta_map_heatmap.pdf
figures/duration_curve.pdf
figures/multi_moment_curve.pdf
```

程序应对每个桶验证：

- GT 和 prediction 的 qid 完整对应；
- 正例定位桶不错误计算 AUROC；
- mR+/mIoU+ 只在至少两个 GT 的查询上计算；
- 单类别存在性桶不输出无意义 AUROC；
- 每个结果包含分桶样本数、正例数和负例数；
- 分桶之间的重叠关系被明确记录。

## 11. 预期可形成的研究发现

完成 Seed-2023 配对分桶和 bootstrap 置信区间评测后，优先验证以下可证伪结论：

- **Finding 1：** LS-DQ-CGP 的主要收益来自中等和较长目标的候选排序，而非极短目标；
- **Finding 2：** 完整 late-semantic 路径提升总体定位，但没有提升存在性分类，甚至可能弱于 Native baseline；
- **Finding 3：** RCG/BPS/FRF 的贡献随目标数量、时间跨度和动作语义变化，而不是在所有查询上均匀生效；
- **Finding 4：** 全局存在性指标受到数据源与查询模板捷径影响，WorldCup 内部评测更能反映真实拒识能力；
- **Finding 5：** 若极短目标的 Oracle Recall@10 同样很低，则未来应优先改进时间分辨率和 span proposal，而不是继续增加语义适配模块。

只有当对应的 Seed-2023 配对差值、95% CI 和样本量共同支持时，才能将上述假设写成正式结论；结论范围应明确限定在本数据划分和随机种子下。
