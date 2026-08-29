#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${DQ_COMPONENT_OUT:-${ROOT}/outputs/dq_cgp_working_part_seed2023}"
PY="${DQ_COMPONENT_PYTHON:-python}"

variants=(baseline full no_inject no_binding no_route injection_only binding_only route_only)
for i in "${!variants[@]}"; do
  variant="${variants[$i]}"
  gpu=$((i % 2))
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -m dq_cgp_working_part_lab.train_variant \
    --variant "${variant}" --seed 2023 --output "${OUT}/${variant}" --device cuda &
  if (( i % 2 == 1 )); then wait; fi
done
wait

for variant in "${variants[@]}"; do
  CUDA_VISIBLE_DEVICES=0 "${PY}" -m dq_cgp_working_part_lab.evaluate_variant \
    --checkpoint "${OUT}/${variant}/best.ckpt" \
    --output "${OUT}/${variant}/test" --split test --device cuda
done
"${PY}" -m dq_cgp_working_part_lab.summarize --root "${OUT}"
