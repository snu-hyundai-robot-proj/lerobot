#!/usr/bin/env bash
# rh56f1 full ablation matrix on top of DINOv2-S + DR.
# Two GPU chains run in parallel; multicam jobs implicitly wait for the multicam
# port (which finishes long before any chain reaches them).
#
# GPU 0 chain (~46h): ACT-sc → DP-FlowMatch-sc → DP-FlowMatch-mc
# GPU 1 chain (~39h): DP-DDIM-sc → ACT-mc → DP-DDIM-mc
set -uo pipefail
source /home/ngseo/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
cd /home/ngseo/remove_hook/lerobot

LOG_DIR=/home/ngseo/remove_hook/lerobot/outputs/train/_logs
OUT_BASE=/home/ngseo/remove_hook/lerobot/outputs/train
SC_ROOT=/home/ngseo/recorded_dataset/lerobot/rh56f1_hookonly
MC_ROOT=/home/ngseo/recorded_dataset/lerobot/rh56f1_hookonly_multicam
SC_REPO=local/rh56f1_hookonly
MC_REPO=local/rh56f1_hookonly_multicam

DR_OPTS=(
  --dataset.image_transforms.enable=true
  --dataset.image_transforms.p_apply=0.5
  --dataset.image_transforms.max_num_transforms=3
  --dataset.image_transforms.domain_randomization=true
)

run_act() {
  local gpu=$1 root=$2 repo=$3 name=$4
  echo "[$(date)] [GPU${gpu}] launching ${name}"
  CUDA_VISIBLE_DEVICES=${gpu} lerobot-train \
    --dataset.repo_id=${repo} \
    --dataset.root=${root} \
    "${DR_OPTS[@]}" \
    --policy.type=act \
    --policy.vision_backbone=dinov2 \
    --policy.dinov2_model_name=facebook/dinov2-small \
    --policy.freeze_vision_backbone=true \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir=${OUT_BASE}/${name} \
    --job_name=${name} \
    --batch_size=8 \
    --steps=800000 \
    --eval_freq=0 \
    > "${LOG_DIR}/${name}.log" 2>&1
  echo "[$(date)] [GPU${gpu}] finished ${name}"
}

run_dp() {
  local gpu=$1 root=$2 repo=$3 name=$4 scheduler=$5  # scheduler in {DDIM, FlowMatch}
  echo "[$(date)] [GPU${gpu}] launching ${name} (scheduler=${scheduler})"
  local extra=()
  if [ "${scheduler}" = "FlowMatch" ]; then
    extra=(--policy.noise_scheduler_type=FlowMatch --policy.num_inference_steps=1)
  fi
  CUDA_VISIBLE_DEVICES=${gpu} lerobot-train \
    --dataset.repo_id=${repo} \
    --dataset.root=${root} \
    "${DR_OPTS[@]}" \
    --policy.type=diffusion \
    --policy.vision_backbone=dinov2 \
    --policy.dinov2_model_name=facebook/dinov2-small \
    --policy.freeze_vision_backbone=true \
    --policy.spatial_softmax_num_keypoints=64 \
    "${extra[@]}" \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir=${OUT_BASE}/${name} \
    --job_name=${name} \
    --batch_size=64 \
    --steps=200000 \
    --eval_freq=0 \
    > "${LOG_DIR}/${name}.log" 2>&1
  echo "[$(date)] [GPU${gpu}] finished ${name}"
}

wait_for_multicam() {
  while [ ! -f "${MC_ROOT}/meta/info.json" ] || [ "$(grep -c '"total_episodes"' "${MC_ROOT}/meta/info.json" 2>/dev/null || echo 0)" -eq 0 ]; do
    echo "[$(date)] waiting for rh56f1 multicam port..."
    sleep 120
  done
  # ensure final write done — port script logs "Wrote N episodes" on completion
  while ! grep -q "Wrote .* episodes" "${LOG_DIR}/port_rh56f1_multicam.log" 2>/dev/null; do
    sleep 60
  done
}

# ---------- GPU 0 chain ----------
(
  run_act 0 "${SC_ROOT}" "${SC_REPO}" rh56f1_act_dinov2s_dr
  run_dp 0 "${SC_ROOT}" "${SC_REPO}" rh56f1_diffusion_dinov2s_flowmatch FlowMatch
  wait_for_multicam
  run_dp 0 "${MC_ROOT}" "${MC_REPO}" rh56f1_diffusion_dinov2s_flowmatch_multicam FlowMatch
  echo "[$(date)] [GPU0] chain done"
) > "${LOG_DIR}/chain_rh56f1_gpu0.log" 2>&1 &
GPU0_PID=$!

# ---------- GPU 1 chain ----------
(
  run_dp 1 "${SC_ROOT}" "${SC_REPO}" rh56f1_diffusion_dinov2s_dr DDIM
  wait_for_multicam
  run_act 1 "${MC_ROOT}" "${MC_REPO}" rh56f1_act_dinov2s_multicam
  run_dp 1 "${MC_ROOT}" "${MC_REPO}" rh56f1_diffusion_dinov2s_multicam DDIM
  echo "[$(date)] [GPU1] chain done"
) > "${LOG_DIR}/chain_rh56f1_gpu1.log" 2>&1 &
GPU1_PID=$!

echo "GPU0 chain pid=${GPU0_PID}"
echo "GPU1 chain pid=${GPU1_PID}"
wait ${GPU0_PID} ${GPU1_PID}
echo "[$(date)] all rh56f1 chains finished."
