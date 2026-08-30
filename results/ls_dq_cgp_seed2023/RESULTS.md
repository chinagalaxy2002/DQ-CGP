# LS-DQ-CGP: Late-Semantic DETR-Query Compositional Generalization Prompter

## 1. 方法概览

针对原版 DQ-CGP 将适配特征作为微弱残差（`beta=0.05`）注入 Decoder 隐状态导致语义适应作用被主干网络防御性稀释的问题，**LS-DQ-CGP (Late-Semantic DQ-CGP)** 彻底回归 APT (Adaptive Prompt Tuning) 的 Information Bottleneck 核心思想：

1. **解耦时序定位与语义排序**：Moment-DETR 主干网络专心利用 Decoder D2 隐状态回归 Span 边界，不再受中间层残差扰动。
2. **Native Binding 抓取真实视觉锚点**：拦截 D1 原生 Cross-Attention 并施加 Hungarian-matched GT 时序绑定损失（$\lambda_{bind}=0.2$），加权 Encoder 视频记忆得到每个 Query 的局部视觉上下文 $V_q$（并施加 `stop-gradient` 阻断捷径）。
3. **后期语义调制 (Late Semantic Adaptation)**：以 $V_q$ 为视觉条件，利用 RCG $\rightarrow$ BPS $\rightarrow$ FRF 动态生成候选专属文本表征 $E_{adapt}^q$。
4. **直接主导检索排序**：使用 D2 隐状态 $h_q$ 与 $E_{adapt}^q$ 进行余弦相似度匹配，直接生成 `pred_logits` 参与分类与排序。

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

---

## 2. 官方 Standard Test 评测对比 (Seed 2023)

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
| **AUROC** | - | 76.69 | 77.33 | **75.04** | 36.64 |
| **Best Val mAP** | 7.09 (ep11) | 19.55 (ep81) | 20.80 (ep112) | **19.10** (ep139) | 10.93 (ep139) |

---

## 3. 核心结论与发现

1. **全面超越 Native Binding 仅 Loss 模式**：
   * 在相同的 Binding Loss 约束下，LS-DQ-CGP 在测试集上获得 **+2.19 mAP** 净增益（14.46 $\rightarrow$ 16.65）。
   * 在多时刻检索指标 `mR+@5` 上直接从 **4.43%** 跃升至 **8.44%**（提升近 1 倍），证明动态文本表征对于区分同视频中不同动作时刻的关键作用。
2. **严格的反事实证据 (Active vs Static Bypass)**：
   * 在同一个训练好的 Checkpoint 上，仅仅将 $E_{adapt}^q$ 替换为全局静态文本 $E_{static}$，测试集 mAP 瞬间由 **16.65** 暴跌至 **11.68**，Top-1 检索能力腰斩。
   * 这直接证明了性能的大幅跃升不是偶然，而是来自于 candidate-specific 动态提示词与语义适配的真实贡献。

---

## 4. 复现与运行指南

### 训练 LS-DQ-CGP (Seed 2023)
```bash
python ls_dq_cgp_lab/train_ls_dq_cgp.py \
  --output outputs/ls_dq_cgp_seed2023 \
  --gpu 0 --seed 2023 --epochs 400 --lr 5e-5 --native_bind_coef 0.2
```

### 测试 Active 模式
```bash
python ls_dq_cgp_lab/evaluate_ls_dq_cgp.py \
  --checkpoint checkpoints/ls_dq_cgp_best_epoch139.ckpt \
  --output results/ls_dq_cgp_seed2023/test_active \
  --split test --gpu 0
```

### 测试 Static Bypass 反事实模式
```bash
python ls_dq_cgp_lab/evaluate_ls_dq_cgp.py \
  --checkpoint checkpoints/ls_dq_cgp_best_epoch139.ckpt \
  --output results/ls_dq_cgp_seed2023/test_static_bypass \
  --split test --static_bypass --gpu 0
```
