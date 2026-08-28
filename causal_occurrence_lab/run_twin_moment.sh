#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${LAB_ROOT}/.." && pwd)"
PYTHON_BIN=${CAUSAL_PYTHON:-python}
TWIN_ROOT=${CAUSAL_TWIN_ROOT:-${REPO_ROOT}/outputs/causal_ablation/twin_moment}

"${PYTHON_BIN}" -m causal_occurrence_lab.build_twin_moment \
  --output-dir "${TWIN_ROOT}" \
  --data-path "${CAUSAL_TWIN_DATA:-${REPO_ROOT}/data/label/Standard/test.jsonl}" \
  --clip-dir "${CAUSAL_CLIP_FEATURES:-${REPO_ROOT}/Soccergmr/clip}" \
  --slowfast-dir "${CAUSAL_SLOWFAST_FEATURES:-${REPO_ROOT}/Soccergmr/slowfast}" \
  --text-dir "${CAUSAL_TEXT_FEATURES:-${REPO_ROOT}/Soccergmr/clip_text}" \
  --max-samples "${CAUSAL_TWIN_SAMPLES:-160}"

printf 'Generated labels: %s/labels.jsonl\n' "${TWIN_ROOT}"
if [[ "${CAUSAL_TWIN_EVAL:-0}" == "1" ]]; then
  BASELINE=${CAUSAL_BASELINE_CHECKPOINT:-/home/guoxiangyu/.local/share/Trash/files/moment_detr_baseline/best.ckpt}
  FULL=${CAUSAL_FULL_CHECKPOINT:-${REPO_ROOT}/checkpoints/dq_cgp_v3_best_epoch86.ckpt}
  for difficulty in exact moderate mixed; do
    for item in baseline full; do
      checkpoint="${BASELINE}"
      mode=baseline
      if [[ "${item}" == "full" ]]; then checkpoint="${FULL}"; mode=dq_active; fi
      if [[ -f "${checkpoint}" ]]; then
        "${PYTHON_BIN}" -m causal_occurrence_lab.analyze_checkpoints \
          --mode "${mode}" --checkpoint "${checkpoint}" \
          --split test --eval-path "${TWIN_ROOT}/labels_${difficulty}.jsonl" \
          --text-features "${TWIN_ROOT}/clip_text" \
          --video-features "${TWIN_ROOT}/clip" "${TWIN_ROOT}/slowfast" \
          --device "${CAUSAL_DEVICE:-cuda}" \
          --output-dir "${TWIN_ROOT}/eval/${difficulty}/${item}"
      fi
    done
  done
fi
printf 'Evaluate with --eval-path %s/labels.jsonl --text-features %s/clip_text --video-features %s/clip %s/slowfast\n' "${TWIN_ROOT}" "${TWIN_ROOT}" "${TWIN_ROOT}" "${TWIN_ROOT}"
