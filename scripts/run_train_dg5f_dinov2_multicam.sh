#!/usr/bin/env bash
# Multi-camera (d405 + zivid) variants of the DINOv2-S + DR ablation.
# Launches 3 trainings in sequence over 2 GPUs:
#   GPU 0:  ACT-DR-multicam       (800k step)  → DP-FlowMatch-DR-multicam (200k step)
#   GPU 1:  DP-DDIM-DR-multicam   (200k step)
# Adjust the loop / commenting below if you only want a subset.
set -euo pipefail
source /home/ngseo/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
cd /home/ngseo/remove_hook/lerobot

DATASET_ROOT=/home/ngseo/recorded_dataset/lerobot/dg5f_hookonly_multicam
REPO_ID=local/dg5f_hookonly_multicam
LOG_DIR=/home/ngseo/remove_hook/lerobot/outputs/train/_logs
mkdir -p "${LOG_DIR}"

DR_OPTS=(
  --dataset.image_transforms.enable=true
  --dataset.image_transforms.p_apply=0.5
  --dataset.image_transforms.max_num_transforms=3
  --dataset.image_transforms.domain_randomization=true
)

# ---------- GPU 0 chain: ACT then FlowMatch ----------
(
  echo "[$(date)] [GPU0] launching dg5f_act_dinov2s_multicam"
  CUDA_VISIBLE_DEVICES=0 lerobot-train \
    --dataset.repo_id=${REPO_ID} \
    --dataset.root=${DATASET_ROOT} \
    "${DR_OPTS[@]}" \
    --policy.type=act \
    --policy.vision_backbone=dinov2 \
    --policy.dinov2_model_name=facebook/dinov2-small \
    --policy.freeze_vision_backbone=true \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir=/home/ngseo/remove_hook/lerobot/outputs/train/dg5f_act_dinov2s_multicam \
    --job_name=dg5f_act_dinov2s_multicam \
    --batch_size=8 \
    --steps=800000 \
    --eval_freq=0 \
    > "${LOG_DIR}/dg5f_act_dinov2s_multicam.log" 2>&1
  echo "[$(date)] [GPU0] act done; launching dg5f_diffusion_dinov2s_flowmatch_multicam"
  CUDA_VISIBLE_DEVICES=0 lerobot-train \
    --dataset.repo_id=${REPO_ID} \
    --dataset.root=${DATASET_ROOT} \
    "${DR_OPTS[@]}" \
    --policy.type=diffusion \
    --policy.vision_backbone=dinov2 \
    --policy.dinov2_model_name=facebook/dinov2-small \
    --policy.freeze_vision_backbone=true \
    --policy.spatial_softmax_num_keypoints=64 \
    --policy.noise_scheduler_type=FlowMatch \
    --policy.num_inference_steps=1 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir=/home/ngseo/remove_hook/lerobot/outputs/train/dg5f_diffusion_dinov2s_flowmatch_multicam \
    --job_name=dg5f_diffusion_dinov2s_flowmatch_multicam \
    --batch_size=64 \
    --steps=200000 \
    --eval_freq=0 \
    > "${LOG_DIR}/dg5f_diffusion_dinov2s_flowmatch_multicam.log" 2>&1
  echo "[$(date)] [GPU0] all done"
) &
GPU0_PID=$!

# ---------- GPU 1: DP-DDIM ----------
(
  echo "[$(date)] [GPU1] launching dg5f_diffusion_dinov2s_multicam (DDIM)"
  CUDA_VISIBLE_DEVICES=1 lerobot-train \
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
    --output_dir=/home/ngseo/remove_hook/lerobot/outputs/train/dg5f_diffusion_dinov2s_multicam \
    --job_name=dg5f_diffusion_dinov2s_multicam \
    --batch_size=64 \
    --steps=200000 \
    --eval_freq=0 \
    > "${LOG_DIR}/dg5f_diffusion_dinov2s_multicam.log" 2>&1
  echo "[$(date)] [GPU1] all done"
) &
GPU1_PID=$!

wait ${GPU0_PID} ${GPU1_PID}
echo "[$(date)] all multicam jobs finished."
