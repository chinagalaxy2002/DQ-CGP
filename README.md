# DQ-CGP: DETR-Query Compositional Generalization Prompter for GMR

本仓库提供 DQ-CGP V3 的完整训练、推理、评测代码以及已经训练好的 checkpoint。DQ-CGP
面向 Generalized Moment Retrieval（GMR）中的多窗口检索：它不再为所有 DETR candidates
生成同一个全局增强文本，而是把每个原生 DETR query 看作一个候选实例，为其生成独立的
temporal context、basis routing 和 adapted feature。

代码基于 Moment-DETR-GMR，并使用 Soccer-GMR Standard split。DQ-CGP 的完整设计说明见
[DESIGN_DQ_CGP.md](experiments/vmr_cgp/DESIGN_DQ_CGP.md)。

## 1. 方法概览

DQ-CGP 插入在 Moment-DETR 的两个 decoder layers 之间：

```text
video/text joint encoder
  -> decoder layer 1: coarse DETR candidates
  -> per-query temporal binding
  -> per-query RCG -> BPS -> FRF
  -> fixed beta residual injection
  -> decoder layer 2: refined candidates
  -> original class/span heads
```

对应关系为：

| APT/CGP 概念 | DQ-CGP 中的实现 |
|---|---|
| object instance | 原生 DETR query |
| local visual feature | candidate-specific temporal context |
| static semantic feature | masked-pooled CLIP query semantic |
| per-instance basis weights | per-candidate RCG routing |
| adapted semantic feature | 注入 decoder state 的 FRF residual |

模型保留原始 10 个 DETR queries、Hungarian matcher、decoder 层数及 span/class heads。真实
CLIP attention mask 只用于 DQ-CGP 私有的 semantic pooling，Moment-DETR encoder 仍使用论文
baseline 的 legacy 32-token 输入语义。

## 2. 已发布 checkpoint 与结果

Checkpoint：

```text
checkpoints/dq_cgp_v3_best_epoch86.ckpt
```

- checkpoint 字段：`epoch=86`
- seed：`2023`
- 固定 residual scale：`beta=0.05`
- 参数 namespace：`query_cgp.*`
- SHA256：`b0f142fedbacea1077ecc82781d206b10ebd82c18cbcf43c44c256b44af4f0bf`

Validation checkpoint-selection 指标：

| Metric | DQ-CGP V3 |
|---|---:|
| MR-full-mAP | 19.02 |
| MR-full-R1@0.5 | 29.02 |
| MR-full-R1@0.7 | 16.08 |
| MR-full-mAP@0.5 | 38.08 |
| MR-full-mAP@0.75 | 20.28 |

Standard test 的完整 GMR 评测：

| Metric | 本地论文原始 Moment-DETR | DQ-CGP V3 |
|---|---:|---:|
| AUROC | 70.25 | 77.33 |
| G-mIoU@1 | 4.97 | 43.25 |
| mAP | 5.34 | 15.51 |
| mR@5 | 8.74 | 22.45 |
| mR+@5 | 0.61 | 7.16 |
| mIoU@5 | 10.23 | 23.56 |
| mIoU+@5 | 0.58 | 6.93 |

详细文件：

- [DQ-CGP test metrics](results/test/metrics.json)
- [DQ-CGP test predictions](results/test/moment_detr_gmr_test_submission.jsonl)
- [baseline test metrics](results/baseline_test_metrics.json)
- [best validation metrics](results/val_best_metrics.json)

### 组件归因实验

仓库同时提供 seed 2023 下的9组受控实验：完整的
`Binding × Route × Injection` 八组因子消融，以及一组不增加可训练参数的原生 decoder
cross-attention binding 验证。代码、训练/验证日志、官方测试结果和结论见
[component attribution report](results/component_attribution_seed2023/RESULTS.md)。

## 3. 环境

本次结果使用：

```text
Python 3.9.0
PyTorch 2.8.0+cu128
NumPy 2.0.2
SciPy 1.13.1
CUDA GPU: NVIDIA RTX 3090
```

建议安装方式：

```bash
conda create -n dq-cgp python=3.9 -y
conda activate dq-cgp

# 按本机 CUDA 版本安装 PyTorch，然后安装其余依赖
pip install -r requirements.txt
```

## 4. Soccer-GMR 特征

由于 Soccer-GMR 采用 gated access、NDA 和禁止再分发条款，本仓库不包含视频或预计算特征。
请从 Soccer-GMR 官方 gated release 获取 `feature.tar`，并整理为：

```text
Soccergmr/
├── clip/
│   └── <video_id>.npz          # key: features
├── slowfast/
│   └── <video_id>.npz          # key: features
└── clip_text/
    └── qid<query_id>.npz       # keys: last_hidden_state, attention_mask
```

Standard split labels 已包含在：

```text
data/label/Standard/train.jsonl
data/label/Standard/val.jsonl
data/label/Standard/test.jsonl
```

训练前可确认三类特征目录存在：

```bash
test -d Soccergmr/clip
test -d Soccergmr/slowfast
test -d Soccergmr/clip_text
```

## 5. 从头训练 DQ-CGP V3

默认配置：seed 2023、batch size 8、最多 400 epochs、validation patience 50，并使用
positive validation `MR-full-mAP` 保存最佳 checkpoint。

```bash
DQ_PYTHON=python \
DQ_GPU=0 \
DQ_EPOCHS=400 \
DQ_OUTPUT=outputs/dq_cgp_v3_seed2023 \
bash scripts/train_dq_cgp_v3.sh
```

若特征不在默认目录，可覆盖：

```bash
DQ_TEXT_FEATURES=/path/to/clip_text \
DQ_CLIP_FEATURES=/path/to/clip \
DQ_SLOWFAST_FEATURES=/path/to/slowfast \
bash scripts/train_dq_cgp_v3.sh
```

主要训练配置位于：

```text
configs/moment_detr_gmr/model/moment_detr_vmr_cgp_v3.yml
configs/moment_detr_gmr/base.yml
```

训练输出包括：

```text
outputs/dq_cgp_v3_seed2023/
├── best.ckpt
├── train.log
├── val.log
├── best_soccer_gmr_val_preds.jsonl
└── best_soccer_gmr_val_preds_metrics.json
```

## 6. 使用已发布 checkpoint 测试

运行 Standard test：

```bash
DQ_PYTHON=python \
DQ_GPU=0 \
DQ_SPLIT=test \
DQ_CHECKPOINT=checkpoints/dq_cgp_v3_best_epoch86.ckpt \
DQ_OUTPUT=outputs/dq_cgp_v3_test \
bash scripts/evaluate_dq_cgp_v3.sh
```

输出：

```text
outputs/dq_cgp_v3_test/
├── moment_detr_gmr_test_submission.jsonl
└── metrics.json
```

在 full validation（包含 positive/null pairs）上运行同一评测：

```bash
DQ_SPLIT=val \
DQ_OUTPUT=outputs/dq_cgp_v3_full_val \
bash scripts/evaluate_dq_cgp_v3.sh
```

CPU 推理可设置：

```bash
DQ_DEVICE=cpu DQ_SPLIT=test bash scripts/evaluate_dq_cgp_v3.sh
```

## 7. 同 checkpoint 的 beta-zero 反事实

以下命令关闭 DQ-CGP 对 decoder state 的注入，但保持 checkpoint 中所有公共参数不变：

```bash
DQ_SPLIT=test \
DQ_ABLATION=beta_zero \
DQ_OUTPUT=outputs/dq_cgp_v3_test_beta_zero \
bash scripts/evaluate_dq_cgp_v3.sh
```

它用于判断推理期 candidate-specific CGP 路径是否确实影响最终预测，不是另训一个 baseline。

## 8. 测试代码

运行 DQ-CGP 与 VMR-CGP 回归测试：

```bash
python -m unittest \
  experiments.vmr_cgp.tests.test_query_cgp \
  experiments.vmr_cgp.tests.test_vmr_cgp \
  -v
```

核心测试覆盖：

- temporal attention 的 shape、padding mask 与归一化；
- 不同 DETR queries 的 candidate-specific context/routing；
- `beta=0` 与原 Moment-DETR 输出严格等价；
- baseline 到 V3 checkpoint 兼容加载；
- Hungarian matched binding loss；
- 原 DETR loss 到 temporal binding、RCG、BPS 与 FRF 的非零梯度。

校验 checkpoint：

```bash
cd checkpoints
sha256sum -c SHA256SUMS
```

## 9. 代码结构

```text
experiments/vmr_cgp/query_cgp.py          # DQ-CGP: binding, RCG, BPS, FRF
experiments/vmr_cgp/query_checkpoint.py   # V3 checkpoint restore/compatibility
experiments/vmr_cgp/query_ablation.py     # beta-zero inference ablation
models/moment_detr_gmr/moment_detr.py     # model and matched binding loss
models/moment_detr_gmr/moment_transformer.py # decoder inter-layer hook
training/moment_detr_gmr/dataset.py       # private CLIP semantic-mask path
training/moment_detr_gmr/train.py         # training entry point
training/moment_detr_gmr/evaluate.py      # inference entry point
```

## 10. 可复现性说明

- 本地论文原始 Moment-DETR baseline 与 DQ-CGP 使用相同 Standard split、离线特征、seed、
  optimizer 和 checkpoint-selection metric。
- DQ-CGP 不修改 encoder 的 legacy query mask、DETR query 数量、matcher、head 或 decoder 层数。
- 不同 CUDA/PyTorch 版本可能造成小幅数值波动。
- Validation 用于模型选择；test 指标由发布 checkpoint 单独运行得到。

## Citation

Soccer-GMR / GMR benchmark：

```bibtex
@article{ding2026retrieving,
  title={Retrieving Any Relevant Moments: Benchmark and Models for Generalized Moment Retrieval},
  author={Ding, Yiming and Cao, Siyu and Jiao, Luyuan and Li, Yixuan and Wang, Zitong and Liu, Zhiyong and Zhang, Lu},
  journal={arXiv preprint arXiv:2605.02623},
  year={2026}
}
```

## License

代码遵循 [MIT License](LICENSE)。Soccer-GMR 数据、视频与预计算特征遵循其各自的访问协议、
NDA 和版权条款。
