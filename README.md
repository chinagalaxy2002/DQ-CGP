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

---

## 2. 已发布 Checkpoint 与官方评测结果

### Checkpoints
* **LS-DQ-CGP + Existence Head（推荐）**: [Google Drive 下载](https://drive.google.com/open?id=1_ekDDphGKkHm67ovNxz-Y6o-1MULpd8u) (`ls_dq_cgp_exist_best_epoch124.ckpt`, Seed 2023)
* **LS-DQ-CGP (Late Semantic, 无显式 existence head)**: `checkpoints/ls_dq_cgp_best_epoch139.ckpt` (Seed 2023)
* **DQ-CGP V3 (Intermediate Residual)**: `checkpoints/dq_cgp_v3_best_epoch86.ckpt` (Seed 2023)

### Standard Test 官方 GMR 评测对比 (Seed 2023)

| 评估指标 | Baseline | Native Binding | DQ-CGPv3 | LS-DQ-CGP（无 Exist） | **LS-DQ-CGP + Exist** |
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

本次 LS-DQ-CGP + Exist 在 Test mAP 上达到 **18.07**，相对 DQ-CGPv3 提高 0.35，相对 Native Binding 提高 3.61；同时取得最高的 mR@5、mIoU@1/5 和 mIoU+@3/5。DQ-CGPv3 仍在 mR+@5 和 AUROC 上领先。所有结果均为单 seed，0.35 mAP 的差距尚需多 seed 验证。

同 checkpoint 反事实评测中，Active / Static Bypass / Context Roll 的 mAP 分别为 **18.07 / 11.53 / 17.16**；三者 AUROC 均为 75.83，说明 existence 判别与语义排序消融已经解耦。

### 2.1 LS-DQ-CGP + QAP：Query-conditioned Attentive Prompt Pooling（探索性版本）

QAP 保留原有 `Bind → Adapt → Match` 路径，仅将 BPS 的 6-token mean pooling 替换为由 $[V_q;E_{static}]$ 条件化的 attention pooling。该实现采用 baseline-preserving initialization：$W_Q=0$、$b_Q=0$、$W_V=I$，因此训练初始时严格退化为原始 MeanPool；同时提供 `uniform_prompt_pool` 反事实，用于区分 attentive composition 与额外 Q/K/V 参数的贡献。该组合是 factorized composition：**Basis Routing × Prompt Position Attention**，不是对 $16\times6$ 个 basis tokens 独立路由。

当前 Seed 2023 结果如下。QAP 最佳验证 checkpoint 为 Epoch 135，训练在 Epoch 185 early stop。

| 方法 | Test mAP | mR@1 | mR@3 | mR@5 | mIoU@1 | G-mIoU@1 | AUROC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原 LS-DQ-CGP + Exist | **18.07** | **12.35** | **18.74** | **24.49** | **30.03** | **32.39** | **75.83** |
| QAP Active | 14.49 | 9.80 | 16.42 | 19.70 | 26.05 | 30.58 | 74.98 |
| QAP UniformPrompt | 14.49 | 9.80 | 16.42 | 19.70 | 26.05 | 30.58 | 74.98 |

QAP 的最佳验证 MR-full-mAP 为 **20.23**（原 LS-DQ-CGP 为 19.99），但该小幅验证集提升没有迁移到 Test；Active 与 UniformPrompt 的全部官方指标完全一致。因而当前实验不支持 attentive prompt composition 作为主方法，建议将其保留为负结果/消融。代码位于 `ls_dq_cgp_tap_lab/`，训练输出和 checkpoint 默认写入 `outputs/ls_dq_cgp_tap_exist_seed2023/`（不纳入 Git）。

> 指标口径说明：DQ-CGPv3 列统一采用 reproduced checkpoint 的配套结果（mAP 17.72、AUROC 76.23）。released checkpoint 的另一套结果是 mAP 15.51、AUROC 77.33，不再跨 checkpoint 拼接展示。

> 📌 **详细评测文件**：
> - [LS-DQ-CGP + Existence Head 完整结果与反事实分析](results/ls_dq_cgp_exist_seed2023/RESULTS.md)
> - [LS-DQ-CGP 结果与消融分析报告](results/ls_dq_cgp_seed2023/RESULTS.md)
> - [9组因子归因消融实验报告](results/component_attribution_seed2023/RESULTS.md)
> - [DQ-CGP V3 Test Metrics](results/test/metrics.json)

---

## 3. 环境配置

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

experiments/vmr_cgp/                     # 原版 DQ-CGP V3 代码
models/moment_detr_gmr/                  # Moment-DETR-GMR 主干实现
training/moment_detr_gmr/                # 数据集与基础训练模块
results/ls_dq_cgp_seed2023/              # LS-DQ-CGP 评测日志与指标记录
results/ls_dq_cgp_exist_seed2023/        # Exist 版本完整日志、预测与评测报告
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
