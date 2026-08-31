#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT=${1:?Usage: launch_tmux.sh VARIANT [GPU]}
GPU=${2:-0}
PYTHON=${PYTHON:-/home/guoxiangyu/miniconda3/envs/GMR/bin/python}
SESSION="ls_ablation_${VARIANT}_seed2023"
LOG_DIR="${ROOT}/logs"
LOG_FILE="${LOG_DIR}/${SESSION}.log"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
mkdir -p "${LOG_DIR}"

tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && PYTHON='${PYTHON}' bash ls_dq_cgp_ablation_lab/run_experiment.sh '${VARIANT}' '${GPU}' 2>&1 | tee '${LOG_FILE}'"

echo "Started: ${SESSION}"
echo "Attach:  tmux attach -t ${SESSION}"
echo "Log:     ${LOG_FILE}"

