# Occurrence-binding experiments

This directory is an isolated analysis harness for the DQ-CGP occurrence-binding
claim.  It does not modify files under `models/`, `experiments/`,
`training/`, or `scripts/`.

The harness uses the repository's released model and matcher through imports,
then installs two inference-only runtime wrappers:

* native decoder cross-attention capture for both decoder layers;
* `context_roll`, which permutes the already-computed DQ-CGP context axis and
  recomputes only the downstream pointwise routing/FRF path.

## Quick start

From the repository root:

```bash
bash occurrence_binding_lab/run_occurrence_binding.sh
```

Useful overrides include:

```bash
OBS_SPLIT=val OBS_DEVICE=cuda OBS_NUM_WORKERS=0 \
  bash occurrence_binding_lab/run_occurrence_binding.sh
```

The default baseline checkpoint is the local checkpoint used by the existing
query-ownership analysis.  Set `OBS_BASELINE_CHECKPOINT` if it is elsewhere.

Outputs are written below `occurrence_binding_lab/outputs/` by default.  The
JSONL records contain compact per-query metrics; full attention maps are only
saved when `--save-attention-qids` is supplied to `run_analysis.py`.

