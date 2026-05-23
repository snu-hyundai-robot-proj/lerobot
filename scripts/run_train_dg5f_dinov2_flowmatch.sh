#!/usr/bin/env bash
# dg5f Diffusion Policy with FlowMatch (rectified flow) loss + DINOv2-small + DR.
# This single trained checkpoint serves the 4-way ablation:
#   DP-DDIM        → existing checkpoint at outputs/train/dg5f_diffusion_dinov2s_dr
#   DP-FlowMatch   → this run, eval with --policy.num_inference_steps=1 (or 4)
#   DP-FlowMatch+RTC → same checkpoint, eval with --policy.use_rtc=true
#   DP-AdaFlow     → same checkpoint, eval with --policy.use_adaflow_inference=true
set -euo pipefail
source /home/ngseo/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
cd /home/ngseo/remove_hook/lerobot

DATASET_ROOT=/home/ngseo/recorded_dataset/lerobot/dg5f_hookonly
REPO_ID=local/dg5f_hookonly
LOG_DIR=/home/ngseo/remove_hook/lerobot/outputs/train/_logs
mkdir -p "${LOG_DIR}"

# Same DR setup as the DR ablation run for fair comparison.
DR_OPTS=(
  --dataset.image_transforms.enable=true
  --dataset.image_transforms.p_apply=0.5
  --dataset.image_transforms.max_num_transforms=3
  --dataset.image_transforms.domain_randomization=true
)

echo "[$(date)] launching dg5f_diffusion_dinov2s_flowmatch on GPU 0"
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
  --output_dir=/home/ngseo/remove_hook/lerobot/outputs/train/dg5f_diffusion_dinov2s_flowmatch \
  --job_name=dg5f_diffusion_dinov2s_flowmatch \
  --batch_size=64 \
  --steps=200000 \
  --eval_freq=0 \
  > "${LOG_DIR}/dg5f_diffusion_dinov2s_flowmatch.log" 2>&1 &
DP_PID=$!
echo "  → pid=${DP_PID}"
wait ${DP_PID}
echo "[$(date)] flowmatch training finished."
