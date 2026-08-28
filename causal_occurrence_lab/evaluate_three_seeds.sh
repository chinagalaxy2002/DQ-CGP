#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${LAB_ROOT}/.." && pwd)"
PYTHON_BIN=${CAUSAL_PYTHON:-python}
GPU=${CAUSAL_GPU:-0}
TRAIN_ROOT=${CAUSAL_TRAIN_ROOT:-${REPO_ROOT}/outputs/causal_ablation}
EVAL_ROOT=${CAUSAL_EVAL_ROOT:-${REPO_ROOT}/outputs/causal_ablation/eval_three_seeds/test}
VARIANTS=${CAUSAL_VARIANTS:-baseline,full,no_bind,supervision_only}
TEXT_FEATURES=${CAUSAL_TEXT_FEATURES:-${REPO_ROOT}/Soccergmr/clip_text}
CLIP_FEATURES=${CAUSAL_CLIP_FEATURES:-${REPO_ROOT}/Soccergmr/clip}
SLOWFAST_FEATURES=${CAUSAL_SLOWFAST_FEATURES:-${REPO_ROOT}/Soccergmr/slowfast}

IFS=',' read -r -a variant_list <<< "${VARIANTS}"
for seed in 2023 2024 2025; do
  for variant in "${variant_list[@]}"; do
    checkpoint="${TRAIN_ROOT}/${variant}_seed${seed}/best.ckpt"
    if [[ ! -f "${checkpoint}" ]]; then
      printf 'Skipping %s seed %s; checkpoint not found: %s\n' "${variant}" "${seed}" "${checkpoint}" >&2
      continue
    fi
    mode=dq_active
    if [[ "${variant}" == "baseline" ]]; then mode=baseline; fi
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m causal_occurrence_lab.analyze_checkpoints \
      --mode "${mode}" --checkpoint "${checkpoint}" --split test \
      --output-dir "${EVAL_ROOT}/${variant}_seed${seed}" \
      --text-features "${TEXT_FEATURES}" \
      --video-features "${CLIP_FEATURES}" "${SLOWFAST_FEATURES}" \
      --device "${CAUSAL_DEVICE:-cuda}" --batch-size "${CAUSAL_BATCH_SIZE:-4}" \
      --num-workers "${CAUSAL_NUM_WORKERS:-0}"
  done
done

"${PYTHON_BIN}" -m causal_occurrence_lab.summarize_multiseed \
  --root "${EVAL_ROOT}" --variants "${VARIANTS}" --seeds 2023 2024 2025 \
  --bootstrap "${CAUSAL_BOOTSTRAP:-10000}"
