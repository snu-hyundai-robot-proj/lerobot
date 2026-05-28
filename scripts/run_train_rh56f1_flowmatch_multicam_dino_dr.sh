#!/usr/bin/env bash
# rh56f1_hookonly  +  Diffusion Policy
#   scheduler : FlowMatch (num_inference_steps=1)
#   cams      : d405 + zivid  (multicam)
#   backbone  : dinov2-small (frozen)
#   DR        : image_transforms with domain randomization
#
# usage:
#   ./run_train_rh56f1_flowmatch_multicam_dino_dr.sh smoke   # 2k steps sanity run
#   ./run_train_rh56f1_flowmatch_multicam_dino_dr.sh full    # 200k steps full run

set -euo pipefail

MODE=${1:-smoke}
GPU=${CUDA_VISIBLE_DEVICES:-0}

ROOT=/root/lerobot_data/rh56f1_hookonly_multicam
REPO=local/rh56f1_hookonly_multicam
OUT_BASE=/root/lerobot/outputs/train
LOG_DIR=${OUT_BASE}/_logs
mkdir -p "${LOG_DIR}"

if [ "${MODE}" = "smoke" ]; then
  NAME=rh56f1_diffusion_dinov2s_flowmatch_multicam_dr_smoke
  STEPS=2000
elif [ "${MODE}" = "full" ]; then
  NAME=rh56f1_diffusion_dinov2s_flowmatch_multicam_dr
  STEPS=200000
else
  echo "unknown mode: ${MODE}  (use 'smoke' or 'full')" >&2
  exit 1
fi

echo "[$(date)] launching ${NAME} on GPU${GPU} (steps=${STEPS})"

CUDA_VISIBLE_DEVICES=${GPU} lerobot-train \
  --dataset.repo_id=${REPO} \
  --dataset.root=${ROOT} \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.p_apply=0.5 \
  --dataset.image_transforms.max_num_transforms=3 \
  --dataset.image_transforms.domain_randomization=true \
  --policy.type=diffusion \
  --policy.vision_backbone=dinov2 \
  --policy.dinov2_model_name=facebook/dinov2-small \
  --policy.freeze_vision_backbone=true \
  --policy.spatial_softmax_num_keypoints=64 \
  --policy.noise_scheduler_type=FlowMatch \
  --policy.num_inference_steps=1 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=${OUT_BASE}/${NAME} \
  --job_name=${NAME} \
  --batch_size=64 \
  --steps=${STEPS} \
  --eval_freq=0 \
  > "${LOG_DIR}/${NAME}.log" 2>&1

echo "[$(date)] finished ${NAME}"
