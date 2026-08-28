#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${LAB_ROOT}/.." && pwd)"
PYTHON_BIN=${CAUSAL_PYTHON:-python}
RUN_DIR=${1:?usage: run_trajectory.sh RUN_DIR}

"${PYTHON_BIN}" -m causal_occurrence_lab.analyze_trajectory \
  --run-dir "${RUN_DIR}" --device "${CAUSAL_DEVICE:-cuda}" \
  --text-features "${CAUSAL_TEXT_FEATURES:-${REPO_ROOT}/Soccergmr/clip_text}" \
  --video-features "${CAUSAL_CLIP_FEATURES:-${REPO_ROOT}/Soccergmr/clip}" "${CAUSAL_SLOWFAST_FEATURES:-${REPO_ROOT}/Soccergmr/slowfast}" \
  --epochs "${CAUSAL_TRAJECTORY_EPOCHS:-1,5,10,20,40,80,best}"
