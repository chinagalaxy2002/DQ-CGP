# VMR-CGP 研究目标与实验协议

## 1. 唯一主目标

在本地复现的论文原始 MomentDETR-GMR 方法上加入一个面向视频时刻检索（VMR）的
Compositional Generalization Prompter（VMR-CGP），重新训练后使预先指定的主指标优于本地
MomentDETR-GMR。

本阶段的主指标沿用训练代码的 checkpoint 选择指标：Standard validation 上的
`MR-full-mAP`。本地无 CGP 的论文原始 MomentDETR-GMR 已有结果为 `8.59`，因此第一阶段成功条件是：

```text
MR-full-mAP(VMR-CGP) > 8.59
```

这不是要求 AUROC、mR、mR+、mIoU+ 等所有指标同时上升。它们全部保留并报告，用于判断
收益来自哪里以及是否产生明显副作用。由于研究动机与多时刻检索有关，`mR+@5` 和
`mIoU+@5` 是重要的次要指标，但不作为第一阶段的硬性成功门槛。

## 2. 主比较

论文主比较只包含两种方法：

1. `MomentDETR-GMR`：本地从头训练的论文原始方法，不使用任何 CGP 文本增强模块；
2. `MomentDETR-GMR + VMR-CGP`：在相同方法中只加入 VMR-CGP，并重新从头训练。

两次训练使用相同的 Standard train/validation 数据、离线 CLIP/SlowFast 特征、论文原始
legacy query 流程、随机种子、公共参数初始化、优化器、学习率、batch size、epoch 上限、early-stop
规则和 checkpoint 选择指标。关闭残差、替换 prompt 等实验仅属于 VMR-CGP 内部消融，不称为
baseline。

当前固定的本地比较对象为：

```text
experiments/temporal_cgp/runs/original_moment_detr_seed2023/best.ckpt
```

该 checkpoint 是 `seed=2023` 下从头训练的论文原始 `moment_detr`，最佳 epoch 为 57
（zero-based），`use_query_attention_mask=false`，不包含任何 CGP 参数。第一版 VMR-CGP 也保持
`use_query_attention_mask=false`，避免把修复 query mask 的收益混入 CGP 主比较。

## 3. VMR-CGP 的模块边界

VMR-CGP 只增强输入 MomentDETR 的文本 token 特征：

```text
Q_enhanced = VMR_CGP(video_tokens, text_tokens, masks)
[video_tokens; Q_enhanced] -> unchanged MomentDETR encoder/decoder/heads
```

模块必须满足：

- 输入和输出 text tensor 形状相同；
- 不追加额外 normalized query token；
- 不增加 temporal proposal、DETR query、matcher、span head 或 decoder；
- 保留 APT 的 `RCG -> BPS -> FRF` 主链路；
- 初始残差很小，避免在训练开始时破坏原始文本特征；
- 使用所有 GT windows 的并集训练多时刻相关性，而不是只使用第一个窗口。

## 4. 第一版算法

第一版使用 token-frame sigmoid relation，允许同一文本 token 同时响应多个视频时刻：

\[
S_{lt}=\gamma\,\langle \hat W_q q_l,\hat W_v v_t\rangle,
\qquad A_{lt}=\sigma(S_{lt}).
\]

每个文本 token 聚合自己的视觉证据，经 RCG 得到 basis weights：

\[
c_l=\frac{1}{T_{valid}}\sum_t m_t A_{lt}W_cv_t,
\]

\[
w_l=\operatorname{softmax}
\left(\operatorname{MLP}([q_l;c_l;q_l\odot c_l])/\tau\right).
\]

BPS 合成真正的 prompt sequence：

\[
P_l=\sum_i w_{li}B_i,
\qquad B_i\in\mathbb R^{L_p\times D}.
\]

FRF 的更新内容只能从 synthesized prompt 读取，避免 MLP 绕过 BPS：

\[
u_l=\operatorname{Attn}(q_l,P_l,P_l),
\qquad Q_l^{enh}=q_l+\alpha g_l\odot u_l.
\]

第一版默认 `num_basis=16`、`prompt_length=4`。训练使用原 MomentDETR 损失，并增加：

- `loss_vmr_cgp_rel`: 对所有 GT windows 并集的 frame relevance 监督；
- `loss_vmr_cgp_route`: 防止所有样本长期只使用同一 basis 的轻量路由正则。

## 5. 结果记录原则

训练结束后至少记录：

- 最佳 checkpoint epoch 和 `MR-full-mAP`；
- 完整 validation 的 mAP、mR@5、mR+@5、mIoU+@5、AUROC、G-mIoU@5；
- token-frame relevance 在 GT 与 background 上的区分能力；
- basis 使用率、conditional entropy 和 marginal entropy；
- 相对本地 MomentDETR-GMR 的逐项差值。

如果主指标没有超过 `8.59`，第一版判定为未达到目标，依据模块诊断修改 CGP，而不是重新定义
baseline 或选择另一个有利指标。

## 6. 文本增强归因要求

主性能目标与机制归因分开判断。模型超过原始 baseline 即表明完整训练方案有效；若要进一步声称
`RCG -> BPS -> FRF` 生成的文本残差本身有效，还必须对同一个最佳 checkpoint 做 identity
ablation：保持训练得到的公共参数不变，只令 `Q_enhanced=Q`。active checkpoint 应优于该
identity ablation，并同时记录 valid token 上的
`||Q_enhanced-Q||/||Q||`。这样不会把训练期辅助损失的正则化作用误写成推理期特征增强作用。
