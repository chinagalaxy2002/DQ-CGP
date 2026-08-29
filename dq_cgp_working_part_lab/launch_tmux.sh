#!/usr/bin/env bash
set -euo pipefail

SESSION=${DQ_COMPONENT_TMUX_SESSION:-dq_work_seed2023}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT=${DQ_COMPONENT_OUT:-${ROOT}/outputs/dq_cgp_working_part_seed2023}
PY=${DQ_COMPONENT_PYTHON:-python}

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi

mkdir -p "${OUT}/launcher_logs"
variants=(baseline full no_inject no_binding no_route injection_only binding_only route_only)
gpus=(0 1 0 1 0 1 0 1)

for i in "${!variants[@]}"; do
  variant=${variants[$i]}
  gpu=${gpus[$i]}
  command="cd '${ROOT}' && set -o pipefail && CUDA_VISIBLE_DEVICES='${gpu}' '${PY}' -m dq_cgp_working_part_lab.train_variant --variant '${variant}' --seed 2023 --output '${OUT}/${variant}' --device cuda --overwrite 2>&1 | tee '${OUT}/launcher_logs/${variant}.log'; status=\${PIPESTATUS[0]}; echo TRAIN_EXIT_STATUS=\${status}; exec bash"
  if ((i == 0)); then
    tmux new-session -d -s "${SESSION}" -n "${variant}" -c "${ROOT}" "bash -lc \"${command}\""
  else
    tmux new-window -d -t "${SESSION}" -n "${variant}" -c "${ROOT}" "bash -lc \"${command}\""
  fi
done

tmux select-window -t "${SESSION}:baseline"
echo "launched ${#variants[@]} windows in tmux session ${SESSION}"
tmux list-windows -t "${SESSION}" -F '#{window_index} #{window_name} #{pane_pid}'
