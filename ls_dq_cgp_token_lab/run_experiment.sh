#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/guoxiangyu/miniconda3/envs/GMR/bin/python"
GPU=${1:-1}
OUTPUT="${ROOT}/outputs/token_ls_dq_cgp_seed2023"

echo "=========================================================="
echo "Starting Token LS-DQ-CGP Experiment on GPU ${GPU}"
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
echo "Experiment Completed Successfully!"
echo "Results:"
echo "  Active:        ${OUTPUT}/test_active/result.json"
echo "  Token Static:  ${OUTPUT}/test_token_static/result.json"
echo "=========================================================="
