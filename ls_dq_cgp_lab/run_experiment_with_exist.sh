#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=${PYTHON:-python}
GPU=${1:-1}
OUTPUT="${ROOT}/outputs/ls_dq_cgp_exist_seed2023"

echo "=========================================================="
echo "Starting LS-DQ-CGP (With Existence Head) on GPU ${GPU}"
echo "Output Directory: ${OUTPUT}"
echo "=========================================================="

# 1. Run Training
"${PYTHON}" -u "${ROOT}/ls_dq_cgp_lab/train_ls_dq_cgp.py" \
  --output "${OUTPUT}" \
  --gpu "${GPU}" \
  --seed 2023 \
  --epochs 400 \
  --lr 5e-5 \
  --native_bind_coef 0.2 \
  --use_exist_head \
  --overwrite

echo "=========================================================="
echo "Training Complete! Running Active Evaluation on Test Split"
echo "=========================================================="

# 2. Evaluate Active Mode on Test Split
"${PYTHON}" -u "${ROOT}/ls_dq_cgp_lab/evaluate_ls_dq_cgp.py" \
  --checkpoint "${OUTPUT}/best.ckpt" \
  --output "${OUTPUT}/test_active" \
  --split test \
  --gpu "${GPU}"

echo "=========================================================="
echo "Running Counterfactual Static Bypass Evaluation on Test Split"
echo "=========================================================="

# 3. Evaluate Static Bypass Mode on Test Split
"${PYTHON}" -u "${ROOT}/ls_dq_cgp_lab/evaluate_ls_dq_cgp.py" \
  --checkpoint "${OUTPUT}/best.ckpt" \
  --output "${OUTPUT}/test_static_bypass" \
  --split test \
  --static_bypass \
  --gpu "${GPU}"

echo "=========================================================="
echo "Running Counterfactual Context Roll Evaluation on Test Split"
echo "=========================================================="

# 4. Evaluate Context Roll Mode on Test Split
"${PYTHON}" -u "${ROOT}/ls_dq_cgp_lab/evaluate_ls_dq_cgp.py" \
  --checkpoint "${OUTPUT}/best.ckpt" \
  --output "${OUTPUT}/test_context_roll" \
  --split test \
  --context_roll \
  --gpu "${GPU}"

echo "=========================================================="
echo "Experiment with Existence Head Completed Successfully!"
echo "=========================================================="
