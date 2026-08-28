#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${LAB_ROOT}/.." && pwd)"
PYTHON_BIN=${CAUSAL_PYTHON:-python}
SPLIT=${CAUSAL_SPLIT:-test}
DEVICE=${CAUSAL_DEVICE:-cuda}
OUT_ROOT=${CAUSAL_PHASE1_ROOT:-${REPO_ROOT}/outputs/causal_ablation/phase1/${SPLIT}}
DQ_CHECKPOINT=${CAUSAL_DQ_CHECKPOINT:-${REPO_ROOT}/checkpoints/dq_cgp_v3_best_epoch86.ckpt}
BASELINE_CHECKPOINT=${CAUSAL_BASELINE_CHECKPOINT:-/home/guoxiangyu/.local/share/Trash/files/moment_detr_baseline/best.ckpt}
TEXT_FEATURES=${CAUSAL_TEXT_FEATURES:-${REPO_ROOT}/Soccergmr/clip_text}
CLIP_FEATURES=${CAUSAL_CLIP_FEATURES:-${REPO_ROOT}/Soccergmr/clip}
SLOWFAST_FEATURES=${CAUSAL_SLOWFAST_FEATURES:-${REPO_ROOT}/Soccergmr/slowfast}
EVAL_PATH=${CAUSAL_EVAL_PATH:-${REPO_ROOT}/data/label/Standard/${SPLIT}.jsonl}

if [[ ! -f "${BASELINE_CHECKPOINT}" ]]; then
  printf 'Missing baseline checkpoint: %s\n' "${BASELINE_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -f "${DQ_CHECKPOINT}" ]]; then
  printf 'Missing DQ checkpoint: %s\n' "${DQ_CHECKPOINT}" >&2
  exit 2
fi

COMMON=(
  --split "${SPLIT}"
  --eval-path "${EVAL_PATH}"
  --text-features "${TEXT_FEATURES}"
  --video-features "${CLIP_FEATURES}" "${SLOWFAST_FEATURES}"
  --device "${DEVICE}"
  --batch-size "${CAUSAL_BATCH_SIZE:-4}"
  --num-workers "${CAUSAL_NUM_WORKERS:-0}"
  --map-workers "${CAUSAL_MAP_WORKERS:-1}"
)

run_mode() {
  local mode="$1"
  local checkpoint="$2"
  "${PYTHON_BIN}" -m causal_occurrence_lab.analyze_checkpoints \
    --mode "${mode}" --checkpoint "${checkpoint}" \
    --output-dir "${OUT_ROOT}/${mode}" "${COMMON[@]}"
}

run_mode baseline "${BASELINE_CHECKPOINT}"
run_mode dq_active "${DQ_CHECKPOINT}"
run_mode dq_beta_zero "${DQ_CHECKPOINT}"
run_mode dq_stripped "${DQ_CHECKPOINT}"

"${PYTHON_BIN}" -m causal_occurrence_lab.compare_runs \
  --root "${OUT_ROOT}" \
  --runs baseline,dq_active,dq_beta_zero,dq_stripped \
  --baseline baseline --active dq_active --identity dq_beta_zero \
  --stripped dq_stripped \
  --bootstrap "${CAUSAL_BOOTSTRAP:-10000}"

"${PYTHON_BIN}" -m causal_occurrence_lab.verify_strip_equivalence \
  --checkpoint "${DQ_CHECKPOINT}" --split "${SPLIT}" --eval-path "${EVAL_PATH}" \
  --text-features "${TEXT_FEATURES}" --video-features "${CLIP_FEATURES}" "${SLOWFAST_FEATURES}" \
  --device "${DEVICE}" --output "${OUT_ROOT}/strip_equivalence.json"

printf 'Phase-1 analysis written to %s\n' "${OUT_ROOT}"
