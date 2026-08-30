# DQ-CGP: DETR-Query Compositional Generalization Prompter for GMR

本仓库提供 **DQ-CGP V3** 与最新突破性架构 **LS-DQ-CGP (Late-Semantic DQ-CGP)** 的完整训练、推理、评测代码以及训练好的 checkpoint。DQ-CGP 面向 Generalized Moment Retrieval（GMR）中的多窗口检索：它不再为所有 DETR candidates 生成同一个全局增强文本，而是把每个原生 DETR query 看作一个候选实例，为其生成独立的 temporal context、basis routing 和 adapted feature。

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

---

## 2. 已发布 Checkpoint 与官方评测结果

### Checkpoints
* **LS-DQ-CGP (Late Semantic)**: `checkpoints/ls_dq_cgp_best_epoch139.ckpt` (Seed 2023)
* **DQ-CGP V3 (Intermediate Residual)**: `checkpoints/dq_cgp_v3_best_epoch86.ckpt` (Seed 2023)

### Standard Test 官方 GMR 评测对比 (Seed 2023)

| 评估指标 (Metric) | Baseline | Native Binding (零额外参数) | 原版 Full DQ-CGP (中间残差) | **LS-DQ-CGP (Active 模式)** | **同模型 Static Bypass (消融反事实)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Test mAP** | 6.14 | 14.46 | 17.72 (带 exist head) | **16.65** | 11.68 |
| **mR@1** | - | 9.90 | - | **10.84** | 4.95 |
| **mR@3** | - | 16.13 | - | **18.08** | 12.15 |
| **mR@5** | 7.89 | 19.77 | 23.59 | **21.78** | 18.64 |
| **mR+@3 (多时刻)** | - | 1.64 | - | **3.93** (+139.6%) | 1.84 |
| **mR+@5 (多时刻)** | 0.69 | 4.43 | 10.20 | **8.44** (+90.5%) | 6.77 |
| **mIoU@1** | - | 25.42 | - | **25.97** | 12.29 |
| **mIoU@5** | 11.61 | 24.19 | 26.29 | **24.07** | 11.26 |
| **mIoU+@3 (多时刻)**| - | 6.80 | - | **7.35** | 2.29 |
| **mIoU+@5 (多时刻)**| - | 6.21 | - | **7.12** | 2.39 |
| **AUROC** | 70.25 | 76.69 | 77.33 | **75.04** | 36.64 |
| **Best Val mAP** | 7.09 (ep11) | 19.55 (ep81) | 20.80 (ep112) | **19.10** (ep139) | 10.93 (ep139) |

> 📌 **详细评测文件**：
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

### 4.1 训练 LS-DQ-CGP (推荐)
```bash
python ls_dq_cgp_lab/train_ls_dq_cgp.py \
  --output outputs/ls_dq_cgp_seed2023 \
  --gpu 0 \
  --seed 2023 \
  --epochs 400 \
  --lr 5e-5 \
  --native_bind_coef 0.2
```

### 4.2 评测与反事实消融 (Active vs Static Bypass)
```bash
# 1. 运行 Active 模式测试
python ls_dq_cgp_lab/evaluate_ls_dq_cgp.py \
  --checkpoint checkpoints/ls_dq_cgp_best_epoch139.ckpt \
  --output outputs/ls_dq_cgp_test_active \
  --split test \
  --gpu 0

# 2. 运行 Static Bypass 模式反事实测试
python ls_dq_cgp_lab/evaluate_ls_dq_cgp.py \
  --checkpoint checkpoints/ls_dq_cgp_best_epoch139.ckpt \
  --output outputs/ls_dq_cgp_test_bypass \
  --split test \
  --static_bypass \
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
└── test_ls_dq_cgp.py                    # 单元回归测试

experiments/vmr_cgp/                     # 原版 DQ-CGP V3 代码
models/moment_detr_gmr/                  # Moment-DETR-GMR 主干实现
training/moment_detr_gmr/                # 数据集与基础训练模块
results/ls_dq_cgp_seed2023/              # LS-DQ-CGP 评测日志与指标记录
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
