#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${LAB_ROOT}/.." && pwd)"
PYTHON_BIN=${OBS_PYTHON:-python}
SPLIT=${OBS_SPLIT:-test}
DEVICE=${OBS_DEVICE:-cuda}
BATCH_SIZE=${OBS_BATCH_SIZE:-4}
NUM_WORKERS=${OBS_NUM_WORKERS:-0}
OUTPUT_ROOT=${OBS_OUTPUT_ROOT:-${LAB_ROOT}/outputs/occurrence_binding/${SPLIT}}
DQ_CHECKPOINT=${OBS_DQ_CHECKPOINT:-${REPO_ROOT}/checkpoints/dq_cgp_v3_best_epoch86.ckpt}
BASELINE_CHECKPOINT=${OBS_BASELINE_CHECKPOINT:-/home/guoxiangyu/.local/share/Trash/files/moment_detr_baseline/best.ckpt}
TEXT_FEATURES=${OBS_TEXT_FEATURES:-${REPO_ROOT}/Soccergmr/clip_text}
CLIP_FEATURES=${OBS_CLIP_FEATURES:-${REPO_ROOT}/Soccergmr/clip}
SLOWFAST_FEATURES=${OBS_SLOWFAST_FEATURES:-${REPO_ROOT}/Soccergmr/slowfast}
EVAL_PATH=${OBS_EVAL_PATH:-${REPO_ROOT}/data/label/Standard/${SPLIT}.jsonl}
BOOTSTRAP=${OBS_BOOTSTRAP:-10000}

COMMON_ARGS=(
  --split "${SPLIT}"
  --eval-path "${EVAL_PATH}"
  --text-features "${TEXT_FEATURES}"
  --video-features "${CLIP_FEATURES}" "${SLOWFAST_FEATURES}"
  --device "${DEVICE}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
)

run_mode() {
  local mode="$1"
  local checkpoint="$2"
  "${PYTHON_BIN}" -m occurrence_binding.run_analysis \
    --mode "${mode}" \
    --checkpoint "${checkpoint}" \
    --output-dir "${OUTPUT_ROOT}/${mode}" \
    "${COMMON_ARGS[@]}"
}

if [[ ! -f "${BASELINE_CHECKPOINT}" ]]; then
  printf 'Baseline checkpoint not found: %s\nSet OBS_BASELINE_CHECKPOINT to run baseline comparison.\n' "${BASELINE_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -f "${DQ_CHECKPOINT}" ]]; then
  printf 'DQ checkpoint not found: %s\n' "${DQ_CHECKPOINT}" >&2
  exit 2
fi

export PYTHONPATH="${LAB_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"
run_mode baseline "${BASELINE_CHECKPOINT}"
run_mode dq_active "${DQ_CHECKPOINT}"
run_mode dq_beta_zero "${DQ_CHECKPOINT}"
run_mode dq_context_roll "${DQ_CHECKPOINT}"

"${PYTHON_BIN}" -m occurrence_binding.compare \
  --root "${OUTPUT_ROOT}" \
  --bootstrap "${BOOTSTRAP}"

printf 'Occurrence-binding results: %s\n' "${OUTPUT_ROOT}"

