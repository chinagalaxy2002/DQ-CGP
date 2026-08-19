#!/usr/bin/env bash
set -euo pipefail

DQ_PYTHON=${DQ_PYTHON:-python}
DQ_GPU=${DQ_GPU:-0}
DQ_EPOCHS=${DQ_EPOCHS:-400}
DQ_OUTPUT=${DQ_OUTPUT:-outputs/dq_cgp_v3_seed2023}
DQ_TEXT_FEATURES=${DQ_TEXT_FEATURES:-Soccergmr/clip_text}
DQ_CLIP_FEATURES=${DQ_CLIP_FEATURES:-Soccergmr/clip}
DQ_SLOWFAST_FEATURES=${DQ_SLOWFAST_FEATURES:-Soccergmr/slowfast}

CUDA_VISIBLE_DEVICES="${DQ_GPU}" "${DQ_PYTHON}" \
  training/moment_detr_gmr/train.py \
  --model moment_detr_vmr_cgp_v3 \
  --dataset soccer_gmr \
  --feature clip_slowfast \
  --train_path data/label/Standard/train.jsonl \
  --eval_path data/label/Standard/val.jsonl \
  --t_feat_dir "${DQ_TEXT_FEATURES}" \
  --v_feat_dirs "${DQ_CLIP_FEATURES}" "${DQ_SLOWFAST_FEATURES}" \
  --results_dir "${DQ_OUTPUT}" \
  --n_epoch "${DQ_EPOCHS}"
