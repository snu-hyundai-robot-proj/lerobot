#!/usr/bin/env bash
# dg5f DP + ACT with frozen DINOv2-small backbone AND online domain randomization.
#
# Domain randomization is applied to ~50% of samples (p_apply=0.5). Each augmented
# sample gets up to `max_num_transforms` transforms sampled by weight from:
#   color jitter (brightness/contrast/saturation/hue) + sharpness + affine
#   + RandomLighting + RandomGamma + RandomSensorNoise + GaussianBlur
#
# Output dirs use the suffix `_dr` to keep them separate from the non-DR runs.
set -euo pipefail
source /home/ngseo/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
cd /home/ngseo/remove_hook/lerobot

DATASET_ROOT=/home/ngseo/recorded_dataset/lerobot/dg5f_hookonly
REPO_ID=local/dg5f_hookonly
LOG_DIR=/home/ngseo/remove_hook/lerobot/outputs/train/_logs
mkdir -p "${LOG_DIR}"

# Common DR overrides — passed to both DP and ACT runs.
# `domain_randomization=true` triggers __post_init__ to bump the cost-weighted
# DR weights (lighting, gamma, sensor_noise, gaussian_blur, corner_crop,
# intrinsics, unprocessor, corruption). p_apply=0.5 → augment 50% of samples.
DR_OPTS=(
  --dataset.image_transforms.enable=true
  --dataset.image_transforms.p_apply=0.5
  --dataset.image_transforms.max_num_transforms=3
  --dataset.image_transforms.domain_randomization=true
)

echo "[$(date)] launching dg5f_diffusion_dinov2s_dr on GPU 0"
CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --dataset.repo_id=${REPO_ID} \
  --dataset.root=${DATASET_ROOT} \
  "${DR_OPTS[@]}" \
  --policy.type=diffusion \
  --policy.vision_backbone=dinov2 \
  --policy.dinov2_model_name=facebook/dinov2-small \
  --policy.freeze_vision_backbone=true \
  --policy.spatial_softmax_num_keypoints=64 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=/home/ngseo/remove_hook/lerobot/outputs/train/dg5f_diffusion_dinov2s_dr \
  --job_name=dg5f_diffusion_dinov2s_dr \
  --batch_size=64 \
  --steps=200000 \
  --eval_freq=0 \
  > "${LOG_DIR}/dg5f_diffusion_dinov2s_dr.log" 2>&1 &
DP_PID=$!
echo "  → pid=${DP_PID}"

echo "[$(date)] launching dg5f_act_dinov2s_dr on GPU 1"
CUDA_VISIBLE_DEVICES=1 lerobot-train \
  --dataset.repo_id=${REPO_ID} \
  --dataset.root=${DATASET_ROOT} \
  "${DR_OPTS[@]}" \
  --policy.type=act \
  --policy.vision_backbone=dinov2 \
  --policy.dinov2_model_name=facebook/dinov2-small \
  --policy.freeze_vision_backbone=true \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=/home/ngseo/remove_hook/lerobot/outputs/train/dg5f_act_dinov2s_dr \
  --job_name=dg5f_act_dinov2s_dr \
  --batch_size=8 \
  --steps=800000 \
  --eval_freq=0 \
  > "${LOG_DIR}/dg5f_act_dinov2s_dr.log" 2>&1 &
ACT_PID=$!
echo "  → pid=${ACT_PID}"

echo "[$(date)] both jobs launched. waiting..."
wait ${DP_PID} ${ACT_PID}
echo "[$(date)] both jobs finished."
