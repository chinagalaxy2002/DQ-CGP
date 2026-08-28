#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${LAB_ROOT}/.." && pwd)"
PYTHON_BIN=${CAUSAL_PYTHON:-python}
GPU=${CAUSAL_GPU:-0}
SEED=${CAUSAL_SEED:-2023}
OUT_ROOT=${CAUSAL_TRAIN_ROOT:-${REPO_ROOT}/outputs/causal_ablation}
TEXT_FEATURES=${CAUSAL_TEXT_FEATURES:-${REPO_ROOT}/Soccergmr/clip_text}
CLIP_FEATURES=${CAUSAL_CLIP_FEATURES:-${REPO_ROOT}/Soccergmr/clip}
SLOWFAST_FEATURES=${CAUSAL_SLOWFAST_FEATURES:-${REPO_ROOT}/Soccergmr/slowfast}

# This is a fresh causal-harness reproduction of the released Full-DQ
# settings.  It intentionally does not resume from the released checkpoint.
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m causal_occurrence_lab.train_causal \
  --variant full_repro --seed "${SEED}" \
  --results-dir "${OUT_ROOT}/full_repro_seed${SEED}" \
  --text-features "${TEXT_FEATURES}" \
  --video-features "${CLIP_FEATURES}" "${SLOWFAST_FEATURES}" \
  --device "${CAUSAL_DEVICE:-cuda}"

printf 'Full causal-harness reproduction written to %s\n' \
  "${OUT_ROOT}/full_repro_seed${SEED}"
