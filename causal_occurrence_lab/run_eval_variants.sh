#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${LAB_ROOT}/.." && pwd)"
PYTHON_BIN=${CAUSAL_PYTHON:-python}
SPLIT=${CAUSAL_SPLIT:-test}
DEVICE=${CAUSAL_DEVICE:-cuda}
OUT_ROOT=${CAUSAL_EVAL_ROOT:-${REPO_ROOT}/outputs/causal_ablation/eval/${SPLIT}}
TRAIN_ROOT=${CAUSAL_TRAIN_ROOT:-${REPO_ROOT}/outputs/causal_ablation}
BASELINE_CHECKPOINT=${CAUSAL_BASELINE_CHECKPOINT:-/home/guoxiangyu/.local/share/Trash/files/moment_detr_baseline/best.ckpt}
FULL_CHECKPOINT=${CAUSAL_FULL_CHECKPOINT:-${REPO_ROOT}/checkpoints/dq_cgp_v3_best_epoch86.ckpt}
TEXT_FEATURES=${CAUSAL_TEXT_FEATURES:-${REPO_ROOT}/Soccergmr/clip_text}
CLIP_FEATURES=${CAUSAL_CLIP_FEATURES:-${REPO_ROOT}/Soccergmr/clip}
SLOWFAST_FEATURES=${CAUSAL_SLOWFAST_FEATURES:-${REPO_ROOT}/Soccergmr/slowfast}
EVAL_PATH=${CAUSAL_EVAL_PATH:-${REPO_ROOT}/data/label/Standard/${SPLIT}.jsonl}

COMMON=(--split "${SPLIT}" --eval-path "${EVAL_PATH}" --text-features "${TEXT_FEATURES}" --video-features "${CLIP_FEATURES}" "${SLOWFAST_FEATURES}" --device "${DEVICE}" --batch-size "${CAUSAL_BATCH_SIZE:-4}" --num-workers "${CAUSAL_NUM_WORKERS:-0}")

run_one() {
  local name="$1" mode="$2" checkpoint="$3"
  if [[ ! -f "${checkpoint}" ]]; then
    printf 'Skipping %s; checkpoint not found: %s\n' "${name}" "${checkpoint}" >&2
    return 0
  fi
  "${PYTHON_BIN}" -m causal_occurrence_lab.analyze_checkpoints \
    --mode "${mode}" --checkpoint "${checkpoint}" \
    --output-dir "${OUT_ROOT}/${name}" "${COMMON[@]}"
}

run_one baseline baseline "${BASELINE_CHECKPOINT}"
run_one full dq_active "${FULL_CHECKPOINT}"
run_one no_bind dq_active "${TRAIN_ROOT}/no_bind_seed2023/best.ckpt"
run_one supervision_only dq_active "${TRAIN_ROOT}/supervision_only_seed2023/best.ckpt"
run_one union_bind dq_active "${TRAIN_ROOT}/union_bind_seed2023/best.ckpt"

"${PYTHON_BIN}" -m causal_occurrence_lab.compare_runs \
  --root "${OUT_ROOT}" \
  --runs baseline,full,no_bind,supervision_only,union_bind \
  --active full --identity dq_beta_zero --bootstrap "${CAUSAL_BOOTSTRAP:-10000}"
printf 'Variant evaluation written to %s\n' "${OUT_ROOT}"
