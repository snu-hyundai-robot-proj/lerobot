#!/usr/bin/env bash
# Hyundai Uiwang (left) + Diffusion Policy with FlowMatch scheduler — full run.
#   scheduler : FlowMatch (num_inference_steps=1)   <- the project's "flowmatch policy"
#   backbone  : resnet18 (random init, no download)
#   cams      : front_rgb + wrist_rgb, resized 240x320 + random crop 216x288 (train)
#   dataset   : data/lerobot/hyundai_uiwang_left  (131 episodes / 110568 frames, 30Hz)
#
# Full-res 480x640 is ~15h for 200k; resize cuts pixels ~4x for a feasible run.
#
# usage: ./run_train_uiwang_left_flowmatch_full.sh [STEPS]   (default 200000)
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

STEPS=${1:-200000}
NAME=uiwang_left_flowmatch_full

python3 -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/hyundai_uiwang_left \
  --dataset.root=data/lerobot/hyundai_uiwang_left \
  --policy.type=diffusion \
  --policy.noise_scheduler_type=FlowMatch \
  --policy.num_inference_steps=1 \
  --policy.vision_backbone=resnet18 \
  --policy.resize_shape="[240, 320]" \
  --policy.crop_ratio=0.9 \
  --policy.crop_is_random=true \
  --policy.spatial_softmax_num_keypoints=64 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/${NAME} \
  --job_name=${NAME} \
  --batch_size=64 \
  --num_workers=8 \
  --steps="${STEPS}" \
  --save_freq=20000 \
  --eval_freq=0 \
  --log_freq=200 \
  --wandb.enable=false
