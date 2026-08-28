#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${LAB_ROOT}/.." && pwd)"
PYTHON_BIN=${CAUSAL_PYTHON:-python}
GPU=${CAUSAL_GPU:-0}
OUT_ROOT=${CAUSAL_TRAIN_ROOT:-${REPO_ROOT}/outputs/causal_ablation}
VARIANTS=${CAUSAL_VARIANTS:-baseline,full,no_bind,supervision_only}
TEXT_FEATURES=${CAUSAL_TEXT_FEATURES:-${REPO_ROOT}/Soccergmr/clip_text}
CLIP_FEATURES=${CAUSAL_CLIP_FEATURES:-${REPO_ROOT}/Soccergmr/clip}
SLOWFAST_FEATURES=${CAUSAL_SLOWFAST_FEATURES:-${REPO_ROOT}/Soccergmr/slowfast}

IFS=',' read -r -a variant_list <<< "${VARIANTS}"
for seed in 2023 2024 2025; do
  for variant in "${variant_list[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m causal_occurrence_lab.train_causal \
      --variant "${variant}" --seed "${seed}" \
      --results-dir "${OUT_ROOT}/${variant}_seed${seed}" \
      --text-features "${TEXT_FEATURES}" \
      --video-features "${CLIP_FEATURES}" "${SLOWFAST_FEATURES}" \
      --device "${CAUSAL_DEVICE:-cuda}"
  done
done
