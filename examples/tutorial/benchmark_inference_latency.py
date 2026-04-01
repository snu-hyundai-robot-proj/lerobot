#!/usr/bin/env python
"""Measure policy inference latency (input -> action output) on a dataset sample.

Example:
python examples/tutorial/benchmark_inference_latency.py \
  --dataset-repo-id lerobot/svla_so101_pickplace \
  --policy-path /home/hochan/Projects/lerobot/src/lerobot/outputs/train/dp_test/checkpoints/last/pretrained_model \
  --device cuda \
  --iterations 100
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.control_utils import predict_action


def _to_numpy_observation(sample: dict) -> dict[str, np.ndarray]:
    """Extract observation keys and convert them to numpy arrays for inference."""
    obs: dict[str, np.ndarray] = {}
    for key, value in sample.items():
        if not key.startswith("observation"):
            continue

        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            arr = value
        else:
            continue

        if ".images." in key:
            # Normalize shape to HWC because prepare_observation_for_inference expects HWC.
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))

            # Normalize dtype/range to uint8 so the policy utility can scale to [0,1].
            if arr.dtype != np.uint8:
                if np.issubdtype(arr.dtype, np.floating) and np.nanmax(arr) <= 1.0:
                    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.float32, copy=False)

        obs[key] = arr

    if not obs:
        raise ValueError("No observation keys were found in dataset sample.")

    return obs


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LeRobot policy inference latency.")
    parser.add_argument("--dataset-repo-id", required=True, help="Dataset repo id, e.g. lerobot/svla_so101_pickplace")
    parser.add_argument("--policy-path", required=True, help="Path to pretrained_model directory")
    parser.add_argument("--device", default="cuda", help="Inference device: cuda or cpu")
    parser.add_argument("--iterations", type=int, default=100, help="Number of measured iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--sample-index", type=int, default=0, help="Dataset sample index")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Override diffusion denoising steps at inference time (smaller is faster)",
    )
    args = parser.parse_args()

    policy_path = Path(args.policy_path)
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy path does not exist: {policy_path}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    print("[1/5] Loading dataset sample...")
    dataset = LeRobotDataset(repo_id=args.dataset_repo_id, download_videos=True)
    sample = dataset[args.sample_index]
    observation = _to_numpy_observation(sample)
    task = sample.get("task") if isinstance(sample.get("task"), str) else None

    print("[2/5] Loading policy config/checkpoint...")
    cli_overrides = []
    if args.num_inference_steps is not None:
        cli_overrides.append(f"--num_inference_steps={args.num_inference_steps}")

    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path), cli_overrides=cli_overrides)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = args.device

    print("[3/5] Building policy + processors...")
    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        dataset_stats=dataset.meta.stats,
    )
    policy.eval()
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    device = torch.device(args.device)
    use_amp = bool(getattr(policy.config, "use_amp", False))
    cfg_inference_steps = getattr(policy_cfg, "num_inference_steps", None)
    effective_inference_steps = getattr(getattr(policy, "diffusion", None), "num_inference_steps", None)

    print(f"[4/5] Warmup: {args.warmup} iterations")
    for _ in range(args.warmup):
        _ = predict_action(
            observation=observation,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=use_amp,
            task=task,
            robot_type=None,
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    print(f"[5/5] Measuring: {args.iterations} iterations")
    latencies_s: list[float] = []
    action_shape = None

    for _ in range(args.iterations):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        action = predict_action(
            observation=observation,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=use_amp,
            task=task,
            robot_type=None,
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        latencies_s.append(t1 - t0)
        if action_shape is None:
            action_shape = tuple(action.shape)

    mean_s = statistics.mean(latencies_s)
    std_s = statistics.pstdev(latencies_s)
    p50_s = _percentile(latencies_s, 50)
    p95_s = _percentile(latencies_s, 95)
    fps = 1.0 / mean_s if mean_s > 0 else float("inf")

    print("\n=== Inference Latency Report ===")
    print(f"policy_path: {policy_path}")
    print(f"dataset_repo_id: {args.dataset_repo_id}")
    print(f"device: {args.device}")
    print(f"iterations: {args.iterations}")
    print(f"warmup: {args.warmup}")
    print(f"sample_index: {args.sample_index}")
    print(f"configured_num_inference_steps: {cfg_inference_steps}")
    print(f"effective_num_inference_steps: {effective_inference_steps}")
    print(f"action_shape: {action_shape}")
    print(f"avg_latency_ms: {mean_s * 1000:.3f}")
    print(f"std_latency_ms: {std_s * 1000:.3f}")
    print(f"p50_latency_ms: {p50_s * 1000:.3f}")
    print(f"p95_latency_ms: {p95_s * 1000:.3f}")
    print(f"fps_from_avg: {fps:.3f}")


if __name__ == "__main__":
    main()
