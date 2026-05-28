#!/usr/bin/env bash
# dg5f_hookonly  +  Diffusion Policy  (resnet18 image encoder)
#   scheduler : FlowMatch (num_inference_steps=1)
#   cams      : d405 + zivid (multicam)
#   backbone  : torchvision resnet18, GroupNorm (random init, trained along policy)
#   DR        : image_transforms with domain randomization
#   mask      : OFF
#
# Identical recipe to the dinov2-small sibling
# (Ngseo/dg5f_diffusion_dinov2s_flowmatch_multicam_dr) except for the vision encoder.
#
# usage:
#   ./run_train_dg5f_flowmatch_multicam_resnet_dr.sh smoke   # 2k steps sanity
#   ./run_train_dg5f_flowmatch_multicam_resnet_dr.sh full    # 200k steps full

set -euo pipefail

MODE=${1:-smoke}
GPU=${CUDA_VISIBLE_DEVICES:-0}

ROOT=/root/lerobot_data/dg5f_hookonly_multicam
REPO=local/dg5f_hookonly_multicam
OUT_BASE=/root/lerobot/outputs/train
LOG_DIR=${OUT_BASE}/_logs
mkdir -p "${LOG_DIR}"

if [ "${MODE}" = "smoke" ]; then
  NAME=dg5f_diffusion_resnet18_flowmatch_multicam_dr_smoke
  STEPS=2000
elif [ "${MODE}" = "full" ]; then
  NAME=dg5f_diffusion_resnet18_flowmatch_multicam_dr
  STEPS=200000
else
  echo "unknown mode: ${MODE}  (use 'smoke' or 'full')" >&2
  exit 1
fi

echo "[$(date)] launching ${NAME} on GPU${GPU} (steps=${STEPS})"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=${GPU} lerobot-train \
  --dataset.repo_id=${REPO} \
  --dataset.root=${ROOT} \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.p_apply=0.5 \
  --dataset.image_transforms.max_num_transforms=3 \
  --dataset.image_transforms.domain_randomization=true \
  --policy.type=diffusion \
  --policy.vision_backbone=resnet18 \
  --policy.freeze_vision_backbone=false \
  --policy.use_group_norm=true \
  --policy.spatial_softmax_num_keypoints=64 \
  --policy.noise_scheduler_type=FlowMatch \
  --policy.num_inference_steps=1 \
  --policy.use_amp=true \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=${OUT_BASE}/${NAME} \
  --job_name=${NAME} \
  --batch_size=128 \
  --num_workers=32 \
  --steps=${STEPS} \
  --eval_freq=0 \
  > "${LOG_DIR}/${NAME}.log" 2>&1

echo "[$(date)] finished ${NAME}"
