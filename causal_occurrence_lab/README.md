# Causal occurrence-binding lab

这个目录是后续 DQ-CGP 因果拆解实验的隔离 harness。它只在运行时 import 根目录的模型、dataset、matcher 和评估实现；没有修改 `models/`、`training/`、`experiments/`、`configs/` 或现有脚本。

## 实验顺序

第一阶段先分析已有 checkpoint：

```bash
bash causal_occurrence_lab/run_phase1.sh
```

它会运行 `baseline`、`dq_active`、`dq_beta_zero` 和 `dq_stripped`，并输出：

- D1 auxiliary prediction 和 D2 prediction 的 raw-span diagnostic mAP、mR、mR+、mIoU 等 GMR 指标（不应用正式 pipeline 的 `PostProcessorDETR`）；
- D1 own-match、D1 final-match、D2 的 raw/length-normalized AEC、binding margin、ECR；
- 修正后的 one-to-one duplicate attribution rate；
- all multi-occurrence、2 occurrences、>=3 occurrences、clean non-overlapping subset；
- residual 相对更新量、分类/span 改变量、Top-5 query Jaccard；
- conditional/marginal routing entropy、有效 basis 数量和 argmax usage；
- 100 个随机样本的 `dq_beta_zero` 与 stripped model 数值等价性。

默认 baseline checkpoint 是当前机器上的旧路径。如果位置不同，运行前设置：

```bash
CAUSAL_BASELINE_CHECKPOINT=/path/to/baseline/best.ckpt bash causal_occurrence_lab/run_phase1.sh
```

所有路径、设备、split、batch size 和 bootstrap 次数都可通过 `CAUSAL_*` 环境变量覆盖。

## 首轮三次新训练

```bash
bash causal_occurrence_lab/run_first_round.sh
bash causal_occurrence_lab/run_eval_variants.sh
```

训练目录默认为：

```text
outputs/causal_ablation/
├── no_bind_seed2023/
├── supervision_only_seed2023/
└── union_bind_seed2023/
```

`train_causal.py` 的首轮 variant 定义为：

| Variant | binding target | binding coef | route coef | residual injection |
|---|---:|---:|---:|---:|
| `baseline` | none | 0 | 0 | no |
| `full` | matched | 0.2 | 0.01 | yes |
| `no_bind` | matched | 0 | 0.01 | yes |
| `supervision_only` | matched | 0.2 | 0 | no |
| `union_bind` | union | 0.2 | 0.01 | yes |

另外已经实现但默认不首轮运行的控制是：`wrong_bind`、`no_route`、`architecture_only` 和 `native_bind`。其中 `native_bind` 使用 baseline 原生 D1 decoder cross-attention，并施加同样的 matched binding loss，不引入 DQ head。

正式因果训练前先运行完整复现实验：

```bash
bash causal_occurrence_lab/run_full_repro.sh
```

它使用 `full_repro` variant，默认 seed 2023，并写入
`outputs/causal_ablation/full_repro_seed2023/`；这是从头训练的 causal-harness
复现，不会从 release Full checkpoint resume。

训练保持 seed 2023、lr `5e-5`、batch size 8、400 epochs、patience 50，并始终用 validation `MR-full-mAP` 选择 best checkpoint。`--max-train-batches` 仅用于 smoke test，正式实验不要设置。

`query_cgp` route loss 严格复用 production 定义：先把一个 batch 中全部
matched routes `torch.cat`，再计算
`H_conditional - H_marginal`。实现和 production `SetCriterion.loss_query_cgp`
的 binding/route 数值一致性由单元测试覆盖。

## Training trajectory

训练脚本会按 `--trajectory-epochs` 保存 DQ snapshot，并写入 `trajectory_losses.jsonl`：

```bash
bash causal_occurrence_lab/run_trajectory.sh outputs/causal_ablation/supervision_only_seed2023
```

trajectory 分析只使用 val，不使用 test，汇总 `L_bind`、AEC-D1、ECR-D1、AEC-D2、ECR-D2 和 Coverage@5。

## 后续 stress tests

CLIP occurrence similarity 分层：

```bash
bash causal_occurrence_lab/run_similarity.sh
```

Synthetic Twin-Moment 数据生成：

```bash
bash causal_occurrence_lab/run_twin_moment.sh
```

它会在新目录中生成 exact、moderate 和 mixed 三种 repeated-occurrence 数据及其独立 feature files；之后把生成的 `labels.jsonl` 和 feature 目录传给 `analyze_checkpoints.py` 即可评估。

三 seeds 和可选 controls：

```bash
bash causal_occurrence_lab/run_three_seeds.sh
bash causal_occurrence_lab/evaluate_three_seeds.sh
bash causal_occurrence_lab/run_optional_controls.sh
```

`run_three_seeds.sh` 默认包含 baseline、full、NoBind、SupervisionOnly；`evaluate_three_seeds.sh` 随后生成每个 seed 的 records，并由 `summarize_multiseed.py` 输出 mean±std 和 seed-then-qid hierarchical bootstrap。

## 运行检查

```bash
python -m unittest discover -s causal_occurrence_lab -p 'test_*.py' -v
python -m py_compile causal_occurrence_lab/*.py causal_occurrence_lab/tests/*.py
bash -n causal_occurrence_lab/*.sh
```

## 已提交结果

已有 checkpoint 的 phase-1 汇总、逐 qid records、D1/D2 submissions、
stripped 等价性检查、similarity 分层结果和 smoke logs 位于
`causal_occurrence_lab/results/`。其中 `results/RESULTS.md` 区分了完整的
existing-checkpoint 分析和仅用于实现验证的 one-batch smoke run。
