#!/usr/bin/env bash
# GPU 1: rh56f1 ACT -> rh56f1 Diffusion (sequential)
set -euo pipefail
source /home/ngseo/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
cd /home/ngseo/remove_hook/lerobot

export CUDA_VISIBLE_DEVICES=1

DATASET_ROOT=/home/ngseo/recorded_dataset/lerobot/rh56f1_hookonly
REPO_ID=local/rh56f1_hookonly

echo "[$(date)] === rh56f1_act start ==="
lerobot-train \
  --dataset.repo_id=${REPO_ID} \
  --dataset.root=${DATASET_ROOT} \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=/home/ngseo/remove_hook/lerobot/outputs/train/rh56f1_act \
  --job_name=rh56f1_act \
  --batch_size=8 \
  --steps=100000 \
  --eval_freq=0
echo "[$(date)] === rh56f1_act done ==="

echo "[$(date)] === rh56f1_diffusion start ==="
lerobot-train \
  --dataset.repo_id=${REPO_ID} \
  --dataset.root=${DATASET_ROOT} \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=/home/ngseo/remove_hook/lerobot/outputs/train/rh56f1_diffusion \
  --job_name=rh56f1_diffusion \
  --batch_size=64 \
  --steps=100000 \
  --eval_freq=0
echo "[$(date)] === rh56f1_diffusion done ==="
