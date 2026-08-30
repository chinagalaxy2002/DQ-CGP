#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=${PYTHON:-python}
GPU=${1:-1}
OUTPUT="${ROOT}/outputs/token_ls_dq_cgp_exist_seed2023"

echo "=========================================================="
echo "Starting Token LS-DQ-CGP (With Existence Head) on GPU ${GPU}"
echo "Output Directory: ${OUTPUT}"
echo "=========================================================="

# 1. Run Training
"${PYTHON}" -u "${ROOT}/ls_dq_cgp_token_lab/train_ls_dq_cgp.py" \
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
"${PYTHON}" -u "${ROOT}/ls_dq_cgp_token_lab/evaluate_ls_dq_cgp.py" \
  --checkpoint "${OUTPUT}/best.ckpt" \
  --output "${OUTPUT}/test_active" \
  --split test \
  --gpu "${GPU}"

echo "=========================================================="
echo "Running Counterfactual Token Static Evaluation on Test Split"
echo "=========================================================="

# 3. Evaluate Token Static Mode on Test Split
"${PYTHON}" -u "${ROOT}/ls_dq_cgp_token_lab/evaluate_ls_dq_cgp.py" \
  --checkpoint "${OUTPUT}/best.ckpt" \
  --output "${OUTPUT}/test_token_static" \
  --split test \
  --token_static \
  --gpu "${GPU}"

echo "=========================================================="
echo "Running Counterfactual Context Roll Evaluation on Test Split"
echo "=========================================================="

# 4. Evaluate Context Roll Mode on Test Split
"${PYTHON}" -u "${ROOT}/ls_dq_cgp_token_lab/evaluate_ls_dq_cgp.py" \
  --checkpoint "${OUTPUT}/best.ckpt" \
  --output "${OUTPUT}/test_context_roll" \
  --split test \
  --context_roll \
  --gpu "${GPU}"

echo "=========================================================="
echo "Experiment with Existence Head Completed Successfully!"
echo "=========================================================="
