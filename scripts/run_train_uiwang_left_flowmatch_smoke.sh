#!/usr/bin/env bash
# Hyundai Uiwang (left) + Diffusion Policy with FlowMatch scheduler — smoke run.
#   scheduler : FlowMatch (num_inference_steps=1)   <- the project's "flowmatch policy"
#   backbone  : resnet18 (random init, no download — offline friendly)
#   cams      : front_rgb + wrist_rgb (both 640x480)
#   dataset   : data/lerobot/hyundai_uiwang_left  (131 episodes / 110568 frames)
#
# usage: ./run_train_uiwang_left_flowmatch_smoke.sh [STEPS]   (default 200)
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

STEPS=${1:-200}
NAME=uiwang_left_flowmatch_smoke

python3 -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/hyundai_uiwang_left \
  --dataset.root=data/lerobot/hyundai_uiwang_left \
  --policy.type=diffusion \
  --policy.noise_scheduler_type=FlowMatch \
  --policy.num_inference_steps=1 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/${NAME} \
  --job_name=${NAME} \
  --batch_size=16 \
  --num_workers=8 \
  --steps="${STEPS}" \
  --save_freq="${STEPS}" \
  --eval_freq=0 \
  --log_freq=20 \
  --wandb.enable=false
