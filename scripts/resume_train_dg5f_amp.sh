#!/usr/bin/env bash
# Resume dg5f training from last checkpoint with AMP (bf16) enabled.
# Drops updt_s from ~0.255s to ~0.13-0.18s — expected ~1.7-2x throughput.
set -euo pipefail

GPU=${CUDA_VISIBLE_DEVICES:-0}
NAME=dg5f_diffusion_dinov2s_flowmatch_multicam_dr
OUT_BASE=/root/lerobot/outputs/train
OUT_DIR=${OUT_BASE}/${NAME}
LOG_DIR=${OUT_BASE}/_logs

CONFIG_PATH=${OUT_DIR}/checkpoints/last/pretrained_model/train_config.json
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "missing checkpoint config: ${CONFIG_PATH}" >&2
    exit 1
fi

echo "[$(date)] resuming ${NAME} on GPU${GPU} with AMP enabled"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=${GPU} lerobot-train \
  --config_path=${CONFIG_PATH} \
  --resume=true \
  --policy.use_amp=true \
  > "${LOG_DIR}/${NAME}_resume.log" 2>&1

echo "[$(date)] finished ${NAME}"
