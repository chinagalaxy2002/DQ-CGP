#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=${PYTHON:-python}
GPU=${1:-1}
OUTPUT="${OUTPUT:-${ROOT}/outputs/encoder_text_ls_dq_cgp_seed2023}"

echo "=========================================================="
echo "Starting Encoder-Text LS-DQ-CGP Experiment on GPU ${GPU}"
echo "Output Directory: ${OUTPUT}"
echo "=========================================================="

# 1. Run Training
"${PYTHON}" -u "${ROOT}/ls_dq_cgp_encoder_text_lab/train_ls_dq_cgp.py" \
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
"${PYTHON}" -u "${ROOT}/ls_dq_cgp_encoder_text_lab/evaluate_ls_dq_cgp.py" \
  --checkpoint "${OUTPUT}/best.ckpt" \
  --output "${OUTPUT}/test_active" \
  --split test \
  --gpu "${GPU}"

echo "=========================================================="
echo "Running Pre-Encoder Condition Counterfactual on Test Split"
echo "=========================================================="

# 3. Replace E_enc with E_static in both RCG and FRF.
"${PYTHON}" -u "${ROOT}/ls_dq_cgp_encoder_text_lab/evaluate_ls_dq_cgp.py" \
  --checkpoint "${OUTPUT}/best.ckpt" \
  --output "${OUTPUT}/test_pre_encoder_condition" \
  --split test \
  --pre_encoder_condition \
  --gpu "${GPU}"

echo "=========================================================="
echo "Running Context Roll Evaluation on Test Split"
echo "=========================================================="

# 4. Roll V_q for the complete RCG/FRF occurrence path.
"${PYTHON}" -u "${ROOT}/ls_dq_cgp_encoder_text_lab/evaluate_ls_dq_cgp.py" \
  --checkpoint "${OUTPUT}/best.ckpt" \
  --output "${OUTPUT}/test_context_roll" \
  --split test \
  --context_roll \
  --gpu "${GPU}"

echo "=========================================================="
echo "Experiment Completed Successfully!"
echo "Results:"
echo "  Active:        ${OUTPUT}/test_active/result.json"
echo "  Pre-Encoder:   ${OUTPUT}/test_pre_encoder_condition/result.json"
echo "  Context Roll:  ${OUTPUT}/test_context_roll/result.json"
echo "=========================================================="
