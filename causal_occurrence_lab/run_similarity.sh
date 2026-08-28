#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${LAB_ROOT}/.." && pwd)"
ROOT=${CAUSAL_EVAL_ROOT:-${REPO_ROOT}/outputs/causal_ablation/eval/test}
PYTHON_BIN=${CAUSAL_PYTHON:-python}
"${PYTHON_BIN}" -m causal_occurrence_lab.similarity_analysis \
  --root "${ROOT}" --runs "${CAUSAL_SIMILARITY_RUNS:-baseline,full}" \
  --baseline baseline --clip-dir "${CAUSAL_CLIP_FEATURES:-${REPO_ROOT}/Soccergmr/clip}"
