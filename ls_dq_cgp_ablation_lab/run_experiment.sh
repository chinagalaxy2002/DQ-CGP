#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=${PYTHON:-python}
VARIANT=${1:?Usage: run_experiment.sh VARIANT [GPU]}
GPU=${2:-0}
OUTPUT=${ABLATION_OUTPUT:-"${ROOT}/outputs/ls_ablation_${VARIANT}_seed2023"}
OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_ARGS+=(--overwrite)
fi

if [[ "${VARIANT}" == "native_binding_exist_aligned" ]]; then
  "${PYTHON}" -u "${ROOT}/ls_dq_cgp_ablation_lab/train_native_binding_exist.py" \
    --output "${OUTPUT}" --seed 2023 --gpu "${GPU}" --epochs 400 \
    --lr 5e-5 --native_bind_coef 0.2 "${OVERWRITE_ARGS[@]}"
  "${PYTHON}" -u "${ROOT}/ls_dq_cgp_ablation_lab/evaluate_native_binding_exist.py" \
    --checkpoint "${OUTPUT}/best.ckpt" \
    --output "${OUTPUT}/test" --split test --gpu "${GPU}"
else
  case "${VARIANT}" in
    full|rcg_uniform|bps_query_mean|bps_zero|frf_remove) ;;
    *) echo "Unknown variant: ${VARIANT}" >&2; exit 2 ;;
  esac
  "${PYTHON}" -u "${ROOT}/ls_dq_cgp_ablation_lab/train_semantic_ablation.py" \
    --variant "${VARIANT}" --output "${OUTPUT}" --seed 2023 \
    --gpu "${GPU}" --epochs 400 --lr 5e-5 --native_bind_coef 0.2 \
    "${OVERWRITE_ARGS[@]}"
  "${PYTHON}" -u "${ROOT}/ls_dq_cgp_ablation_lab/evaluate_semantic_ablation.py" \
    --variant "${VARIANT}" --checkpoint "${OUTPUT}/best.ckpt" \
    --output "${OUTPUT}/test" --protocol retrained --split test --gpu "${GPU}"
fi

