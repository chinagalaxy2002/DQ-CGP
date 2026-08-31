#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=${PYTHON:-python}
GPU=${1:-0}
OUTPUT=${DELTA_ZERO_OUTPUT:-"${ROOT}/outputs/ls_dq_cgp_delta_zero_train_seed2023"}

echo "Training strict Delta E_q=0 control on GPU ${GPU}"
echo "Output: ${OUTPUT}"

"${PYTHON}" -u "${ROOT}/strict_delta_zero_lab/train_delta_zero.py" \
  --output "${OUTPUT}" \
  --gpu "${GPU}" \
  --seed 2023 \
  --epochs 400 \
  --lr 5e-5 \
  --native_bind_coef 0.2 \
  --overwrite

echo "Training complete; evaluating the best checkpoint on Standard Test"

"${PYTHON}" -u "${ROOT}/strict_delta_zero_lab/evaluate_delta_zero.py" \
  --checkpoint "${OUTPUT}/best.ckpt" \
  --output "${OUTPUT}/test_delta_zero" \
  --split test \
  --gpu "${GPU}"

echo "Strict Delta E_q=0 experiment complete"

