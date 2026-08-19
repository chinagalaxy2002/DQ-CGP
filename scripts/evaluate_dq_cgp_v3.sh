#!/usr/bin/env bash
set -euo pipefail

DQ_PYTHON=${DQ_PYTHON:-python}
DQ_GPU=${DQ_GPU:-0}
DQ_DEVICE=${DQ_DEVICE:-cuda}
DQ_SPLIT=${DQ_SPLIT:-test}
DQ_CHECKPOINT=${DQ_CHECKPOINT:-checkpoints/dq_cgp_v3_best_epoch86.ckpt}
DQ_OUTPUT=${DQ_OUTPUT:-outputs/dq_cgp_v3_${DQ_SPLIT}}
DQ_TEXT_FEATURES=${DQ_TEXT_FEATURES:-Soccergmr/clip_text}
DQ_CLIP_FEATURES=${DQ_CLIP_FEATURES:-Soccergmr/clip}
DQ_SLOWFAST_FEATURES=${DQ_SLOWFAST_FEATURES:-Soccergmr/slowfast}
DQ_ABLATION=${DQ_ABLATION:-}

case "${DQ_SPLIT}" in
  val|test)
    DQ_LABELS="data/label/Standard/${DQ_SPLIT}.jsonl"
    ;;
  *)
    printf 'DQ_SPLIT must be val or test, got: %s\n' "${DQ_SPLIT}" >&2
    exit 2
    ;;
esac

EXTRA_ARGS=()
if [[ -n "${DQ_ABLATION}" ]]; then
  EXTRA_ARGS+=(--query_cgp_ablation "${DQ_ABLATION}")
fi

CUDA_VISIBLE_DEVICES="${DQ_GPU}" "${DQ_PYTHON}" \
  training/moment_detr_gmr/evaluate.py \
  --model moment_detr_vmr_cgp_v3 \
  --dataset soccer_gmr \
  --feature clip_slowfast \
  --model_path "${DQ_CHECKPOINT}" \
  --split test \
  --eval_path "${DQ_LABELS}" \
  --t_feat_dir "${DQ_TEXT_FEATURES}" \
  --v_feat_dirs "${DQ_CLIP_FEATURES}" "${DQ_SLOWFAST_FEATURES}" \
  --results_dir "${DQ_OUTPUT}" \
  --device "${DQ_DEVICE}" \
  "${EXTRA_ARGS[@]}"

"${DQ_PYTHON}" eval/eval_main.py \
  --submission_path "${DQ_OUTPUT}/moment_detr_gmr_test_submission.jsonl" \
  --gt_path "${DQ_LABELS}" \
  --save_path "${DQ_OUTPUT}/metrics.json" \
  --map_num_workers 1
