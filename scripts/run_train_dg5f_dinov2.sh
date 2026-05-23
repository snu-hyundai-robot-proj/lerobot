#!/usr/bin/env bash
# dg5f DP + ACT with frozen DINOv2-small backbone (parallel: DP on GPU 0, ACT on GPU 1).
set -euo pipefail
source /home/ngseo/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
cd /home/ngseo/remove_hook/lerobot

DATASET_ROOT=/home/ngseo/recorded_dataset/lerobot/dg5f_hookonly
REPO_ID=local/dg5f_hookonly
STEPS=200000
LOG_DIR=/home/ngseo/remove_hook/lerobot/outputs/train/_logs
mkdir -p "${LOG_DIR}"

echo "[$(date)] launching dg5f_diffusion_dinov2s on GPU 0"
CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --dataset.repo_id=${REPO_ID} \
  --dataset.root=${DATASET_ROOT} \
  --policy.type=diffusion \
  --policy.vision_backbone=dinov2 \
  --policy.dinov2_model_name=facebook/dinov2-small \
  --policy.freeze_vision_backbone=true \
  --policy.spatial_softmax_num_keypoints=64 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=/home/ngseo/remove_hook/lerobot/outputs/train/dg5f_diffusion_dinov2s \
  --job_name=dg5f_diffusion_dinov2s \
  --batch_size=64 \
  --steps=${STEPS} \
  --eval_freq=0 \
  > "${LOG_DIR}/dg5f_diffusion_dinov2s.log" 2>&1 &
DP_PID=$!
echo "  → pid=${DP_PID}, log=${LOG_DIR}/dg5f_diffusion_dinov2s.log"

echo "[$(date)] launching dg5f_act_dinov2s on GPU 1"
CUDA_VISIBLE_DEVICES=1 lerobot-train \
  --dataset.repo_id=${REPO_ID} \
  --dataset.root=${DATASET_ROOT} \
  --policy.type=act \
  --policy.vision_backbone=dinov2 \
  --policy.dinov2_model_name=facebook/dinov2-small \
  --policy.freeze_vision_backbone=true \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=/home/ngseo/remove_hook/lerobot/outputs/train/dg5f_act_dinov2s \
  --job_name=dg5f_act_dinov2s \
  --batch_size=8 \
  --steps=${STEPS} \
  --eval_freq=0 \
  > "${LOG_DIR}/dg5f_act_dinov2s.log" 2>&1 &
ACT_PID=$!
echo "  → pid=${ACT_PID}, log=${LOG_DIR}/dg5f_act_dinov2s.log"

echo "[$(date)] both jobs launched. waiting..."
wait ${DP_PID} ${ACT_PID}
echo "[$(date)] both jobs finished."
