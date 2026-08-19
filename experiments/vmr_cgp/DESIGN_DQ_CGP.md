# APT 启发下的 DETR-Query CGP 设计

## 1. 从本地 APT 代码能够确认什么

本地 APT 仓库中的 CGP 只定义在：

```text
APT reference implementation (kept outside this release repository)
maskrcnn_benchmark/modeling/adpative_modeling.py
```

该文件给出了一个简洁的 `RCG -> BPS -> FRF` 骨架：

1. RCG：将当前 object 的视觉特征和静态语义特征拼接，生成 basis softmax weights；
2. BPS：用 weights 对共享 basis prompt bank 加权求和，并对 prompt length 平均池化；
3. FRF：拼接 pooled prompt、静态语义和当前 object 的视觉特征，经 MLP 得到 adapted feature。

精确公式为：

\[
w_i=\operatorname{softmax}(\operatorname{MLP}([v_i;e_i])),
\]

\[
P_i=\sum_n w_{i,n}B_n,\qquad p_i=\operatorname{Mean}(P_i),
\]

\[
e_i^{adapt}=\operatorname{MLP}([p_i;e_i;W_vv_i]).
\]

本地文件没有 `alpha`、residual、feature gate、LayerNorm、temperature、temporal attention 或
CGP 专用 loss。更重要的是，它按 object index 分别取 `v_i` 和 `e_i`，每个 object 独立运行
CGP，最后再堆叠 adapted features。

需要注意：该文件没有被本地 APT 正式训练链路 import；`UnifiedAPTFramework` 还引用了未定义的
`APTCoreModule`。因此它应被视为算法参考骨架，不能将这个仓库没有实现的细节声称为 APT
原始代码。

## 2. v1/v2 的根本问题

v1/v2 把 APT 的 instance axis 映射成了 text-token axis：

```text
每个 text token + 整段视频 -> token prompt -> 所有 DETR queries 共享 Q_enhanced
```

这与 APT 的 object-instance conditioning 不同。视频中存在多个相似时刻时，不同 occurrence
会被聚合进共享文本特征，十个 DETR queries 看见的是同一个增强结果。

实验与这个判断一致：

- v1 的 residual scale 塌缩，active 与 identity 相同；
- v2 强制产生约 1.96% 的特征变化，但 active/identity 收益跨 split 不稳定；
- v2 最佳 checkpoint 的 normalized basis entropy 为 0.99945，路由接近均匀。

因此问题不是继续把 `alpha` 调大或调小，而是 CGP 发生在错误的实例粒度上。

## 3. 核心映射

新的主方案命名为：

```text
DETR-Query Compositional Generalization Prompter（DQ-CGP）
```

APT 与 VMR 的映射为：

| APT | DQ-CGP |
|---|---|
| object instance \(i\) | 原有 DETR query/candidate \(j\) |
| object/ROI visual feature \(v_i\) | candidate-specific temporal context \(c_j\) |
| static class semantic \(e_i\) | pooled CLIP query semantic \(e\) |
| per-object basis weights \(w_i\) | per-candidate basis weights \(w_j\) |
| adapted object semantic | adapted moment-query semantic |
| downstream relation head | 原第二层 Moment-DETR decoder 与 span/class heads |

不新增 proposal、DETR query、matcher、span head 或 decoder layer。多实例轴直接复用原模型已有的
10 个 DETR queries。

## 4. 网络流程

### 4.1 原始 joint encoder 与 coarse decoder

保持原 MomentDETR 的输入投影和 joint encoder：

\[
M=\operatorname{Encoder}([V;Q]),
\qquad M_v=M[:,1:T].
\]

第一层原生 decoder 产生 coarse candidate states：

\[
H^{(1)}=\{h_1^{(1)},\ldots,h_M^{(1)}\},
\qquad H^{(1)}\in\mathbb R^{B\times M\times D}.
\]

这里 \(M=10\)。第一层 auxiliary prediction 保持在 CGP 注入前计算，使其仍是原始 coarse
MomentDETR 输出。

### 4.2 每个 candidate 的 temporal binding

使用真实 CLIP attention mask，仅为 CGP 私有语义池化得到：

\[
e=\operatorname{MaskedPool}(Q)\in\mathbb R^{B\times D}.
\]

送入原 MomentDETR encoder 的文本序列和 legacy mask 不改变，因此 `DQ-CGP=off` 时仍能逐元素
恢复原始 baseline。

每个 coarse DETR candidate 独立读取 temporal memory：

\[
z_j=W_h\operatorname{LN}(h_j^{(1)})+W_e e,
\]

\[
s_{j,t}=\frac{z_j^\top W_m\operatorname{LN}(M_{v,t})}{\sqrt D},
\]

\[
A_{j,t}=\operatorname{MaskedSoftmax}_t(s_{j,t}),
\qquad
c_j=\sum_t A_{j,t}W_cM_{v,t}.
\]

输出：

```text
temporal_attention A : [B, 10, T]
temporal_context   C : [B, 10, D]
```

不同 `h_j^(1)` 产生不同 `A_j` 和 `c_j`，从而把多个 occurrence 分开。

### 4.3 按 APT 方式逐 candidate 执行 CGP

RCG：

\[
w_j=\operatorname{softmax}(\operatorname{MLP}_{RCG}([c_j;e])),
\qquad w_j\in\mathbb R^N.
\]

BPS：

\[
P_j=\sum_{n=1}^{N}w_{j,n}B_n,
\qquad
p_j=\operatorname{Mean}(P_j).
\]

FRF：

\[
r_j=\operatorname{MLP}_{FRF}([p_j;e;W_cc_j]).
\]

建议第一版使用与本地 APT 骨架接近的：

```text
num_basis = 16
prompt_length = 6
router_hidden_dim = 256
FRF hidden_dim = 512
softmax temperature = 1
```

### 4.4 注入 refined decoder

APT 直接输出 adapted feature；VMR 中还要保留 coarse moment identity，因此使用固定的小残差注入：

\[
\tilde h_j^{(1)}
=h_j^{(1)}+\beta\operatorname{LN}(W_rr_j),
\qquad \beta=0.05.
\]

`beta` 固定，不学习 gate，避免再次出现 v1 的自动关路问题。第二层原生 decoder 接收
\(\tilde H^{(1)}\)，输出最终类别和窗口：

\[
\tilde H^{(1)}\rightarrow\operatorname{DecoderLayer}_2
\rightarrow(\hat p_j,\hat t_j^s,\hat t_j^e).
\]

完整链路为：

```text
joint encoder
  -> decoder layer 1: coarse moment candidates
  -> per-candidate temporal binding
  -> per-candidate RCG -> BPS -> FRF
  -> decoder layer 2: refined moment candidates
  -> unchanged class/span heads
```

## 5. 多窗口绑定监督

继续使用原 Hungarian matcher。最终 prediction query \(j\) 与 GT window \(k\) 匹配后，对其
temporal attention 施加 candidate-specific binding loss：

\[
\mathcal L_{bind}
=-\frac{1}{|\mathcal M|}
\sum_{(j,k)\in\mathcal M}
\log\left(\sum_{t:\,clip_t\cap GT_k\ne\varnothing}A_{j,t}+\epsilon\right).
\]

这与 v1/v2 的 GT-union loss 不同：

- 不同 matched DETR queries 分别监督到不同 GT windows；
- 不把多个窗口合成一个共享 relevance map；
- unmatched queries 和 null videos 不计算 binding loss，由原 no-object/existence loss 处理；
- 不改变 matcher，也不在推理时使用 GT。

首轮总损失：

\[
\mathcal L
=\mathcal L_{MomentDETR}+0.2\mathcal L_{bind}
+0.01\mathcal L_{route}.
\]

其中 route regularizer 只用于防止 uniform routing：

\[
\mathcal L_{route}
=\mathbb E_{b,j}[H(w_{b,j})]-H(\mathbb E_{b,j}[w_{b,j}]).
\]

最小化它会降低单 candidate 的 routing entropy，同时保持 batch 内 basis 使用多样性。只对
matched positive queries 计算。第一版不再使用 v1/v2 的 GT-union relevance loss。

## 6. 为什么比继续修改 v2 更合理

1. APT 的真正可迁移点是 per-instance adaptation，而不是 residual 公式；
2. 一个视频中的多个 moment 对应多个 DETR candidates，而不是多个文本 token；
3. 每个 candidate 具有不同 temporal context，RCG 才有理由产生不同 basis weights；
4. Hungarian assignment 已经提供 `candidate ↔ GT window` 的一对一监督；
5. CGP 位于两个 decoder layers 之间，能够直接影响最终 span/class prediction；
6. 没有增加检测槽数量，提升不能解释为模型拥有了更多 proposals。

不推荐作为下一步的改法：

- 继续搜索全局 `alpha` 或 gate floor；
- 只把当前 token router 改成 top-k；
- full-video mean pooling 后生成一个共享 prompt；
- 新增另一套 temporal slots，与原有 DETR queries 重复；
- 同时修改 CLIP mask、后处理和 decoder 数量，导致主比较变量失控。

## 7. 只训练一个主方案

主比较仍只有：

```text
本地原始 MomentDETR（复用已有 checkpoint）
vs.
MomentDETR + DQ-CGP（从头训练一次）
```

保持相同 Standard split、离线特征、seed 2023、公共参数初始化、优化器、训练预算和
checkpoint 选择指标。v2 的继续训练属于已有实验，不影响 DQ-CGP 的设计口径。

训练前必须确认：

- adapter disabled 与原 Transformer 输出逐元素一致；
- `beta=0` 与原始 MomentDETR 输出逐元素一致；
- 原 DETR loss 能向 temporal binding、RCG、basis 和 FRF 传递非零梯度；
- 合成双峰输入中两个 queries 能形成不同 temporal attention。

## 8. 最小机制验证

只使用最佳 active checkpoint 做两个推理反事实，不额外训练 baseline：

1. `beta=0`：关闭 DQ-CGP 对 decoder state 的注入；
2. `route-shuffle`：在不同 query/video 间交换 basis weights，但保持 residual 尺度基本不变。

第一阶段目标仍然简单：

\[
mAP_{val}(DQ\text{-}CGP)>8.59.
\]

同时记录：

- active 是否优于同 checkpoint 的 `beta=0`；
- active 是否优于 `route-shuffle`；
- matched queries 的 temporal peak 是否分别落在不同 GT windows；
- mR+@5 和 mIoU+@5 是否改善。

这样可以分别判断：完整方案是否提高性能、推理期 adapted feature 是否有效、以及条件化 basis
composition 是否有效。它们是同 checkpoint 诊断，不是四个训练 baseline。

## 9. 最小代码边界

建议新增独立 namespace，避免误载入 v1/v2：

```text
experiments/vmr_cgp/query_cgp.py        # DQ-CGP 模块
query_cgp.*                             # checkpoint 参数前缀
moment_detr_vmr_cgp_v3.yml              # 独立配置
```

需要修改：

- `moment_transformer.py`：允许在 decoder layer 1 和 2 之间调用可选 adapter；
- `moment_detr.py`：构建 DQ-CGP、传入 text semantic/video memory，并计算 matched binding loss；
- checkpoint/evaluate：恢复 v3 配置及 `beta=0`、`route-shuffle` 推理消融。

保持不变：dataset 标注、DETR query 数量、encoder/decoder 层数、matcher、span/class heads和原
MomentDETR loss。
