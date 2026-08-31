#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=${PYTHON:-python}
GPU=${1:-1}
OUTPUT="${OUTPUT:-${ROOT}/outputs/token_ls_dq_cgp_v2_exist_seed2023}"

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
echo "Running Counterfactual Uniform Text Attention Evaluation on Test Split"
echo "=========================================================="

# 3. Evaluate Uniform Text Attention Mode on Test Split
"${PYTHON}" -u "${ROOT}/ls_dq_cgp_token_lab/evaluate_ls_dq_cgp.py" \
  --checkpoint "${OUTPUT}/best.ckpt" \
  --output "${OUTPUT}/test_uniform_text_attention" \
  --split test \
  --uniform_text_attention \
  --gpu "${GPU}"

echo "=========================================================="
echo "Running Selector-only Context Roll Evaluation on Test Split"
echo "=========================================================="

# 4. Roll V_q only for the occurrence-specific text selector.
"${PYTHON}" -u "${ROOT}/ls_dq_cgp_token_lab/evaluate_ls_dq_cgp.py" \
  --checkpoint "${OUTPUT}/best.ckpt" \
  --output "${OUTPUT}/test_selector_context_roll" \
  --split test \
  --selector_context_roll \
  --gpu "${GPU}"

echo "=========================================================="
echo "Experiment with Existence Head Completed Successfully!"
echo "=========================================================="
