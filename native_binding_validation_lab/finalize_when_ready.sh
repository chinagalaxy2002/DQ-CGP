#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPONENT_ROOT="${ROOT}/outputs/dq_cgp_working_part_seed2023"
NATIVE_ROOT="${ROOT}/outputs/native_binding_validation_seed2023"
variants=(baseline full no_inject no_binding no_route injection_only binding_only route_only)

while pgrep -f 'python -m dq_cgp_working_part_lab.train_variant' >/dev/null \
   || pgrep -f 'python -m native_binding_validation_lab.train_native_binding' >/dev/null; do
  sleep 30
done

for variant in "${variants[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python -m dq_cgp_working_part_lab.evaluate_variant \
    --checkpoint "${COMPONENT_ROOT}/${variant}/best.ckpt" \
    --output "${COMPONENT_ROOT}/${variant}/test" \
    --split test --device cuda
done

CUDA_VISIBLE_DEVICES=0 python -m dq_cgp_working_part_lab.evaluate_variant \
  --checkpoint "${NATIVE_ROOT}/best.ckpt" \
  --output "${NATIVE_ROOT}/test" --split test --device cuda

python -m dq_cgp_working_part_lab.summarize --root "${COMPONENT_ROOT}"

CUDA_VISIBLE_DEVICES=0 python -m native_binding_validation_lab.verify_stripped \
  --checkpoint "${COMPONENT_ROOT}/binding_only/best.ckpt" \
  --output "${NATIVE_ROOT}/binding_only_stripped_equivalence.json" \
  --batches 64

touch "${NATIVE_ROOT}/FINALIZATION_COMPLETE"
echo FINALIZATION_COMPLETE
