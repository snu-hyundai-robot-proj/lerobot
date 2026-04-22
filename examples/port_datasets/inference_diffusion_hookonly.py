"""Run a trained Diffusion Policy checkpoint on the converted hookonly dataset.

Loads the policy + its pre/post-processors from the saved checkpoint, streams observations
from the dataset one-by-one (starting at `--frame-index`), calls `policy.select_action`
for `n_action_steps` steps, and compares the predicted actions to the ground-truth
actions that follow the starting frame.

Example
-------
python examples/port_datasets/inference_diffusion_hookonly.py \
    --checkpoint outputs/train/dg5f_diffusion/checkpoints/last/pretrained_model \
    --dataset-root /home/ngseo/recorded_dataset/lerobot/dg5f_hookonly \
    --frame-index 100
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors


def load_policy(checkpoint: Path, device: str):
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device = device
    policy = DiffusionPolicy.from_pretrained(checkpoint, config=cfg).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(cfg, pretrained_path=str(checkpoint))
    return cfg, policy, preprocessor, postprocessor


def to_batch(sample: dict, device: str) -> dict:
    batch = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(device)
        elif isinstance(v, str):
            batch[k] = [v]
        else:
            batch[k] = v
    return batch


def episode_bounds(ds: LeRobotDataset, frame_index: int) -> tuple[int, int, int]:
    ep_idx = int(ds[frame_index]["episode_index"])
    ep_meta = ds.meta.episodes[ep_idx]
    return ep_idx, int(ep_meta["dataset_from_index"]), int(ep_meta["dataset_to_index"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="path to .../checkpoints/<step>/pretrained_model")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=None,
                        help="how many steps to predict and compare (defaults to cfg.n_action_steps)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--show-first-n", type=int, default=10)
    args = parser.parse_args()

    repo_id = args.repo_id or f"local/{args.dataset_root.name}"
    ds = LeRobotDataset(repo_id, root=args.dataset_root)

    cfg, policy, preprocessor, postprocessor = load_policy(args.checkpoint, args.device)
    n_action = args.num_steps or cfg.n_action_steps
    print(f"loaded Diffusion  n_obs_steps={cfg.n_obs_steps}  horizon={cfg.horizon}  "
          f"n_action_steps={cfg.n_action_steps}  device={args.device}")
    print(f"ckpt={args.checkpoint}")

    ep_idx, ep_from, ep_to = episode_bounds(ds, args.frame_index)
    end = min(args.frame_index + n_action, ep_to)
    n_action = end - args.frame_index
    print(f"frame {args.frame_index}  episode {ep_idx} ({ep_from}..{ep_to})  "
          f"predicting {n_action} steps")

    policy.reset()
    preds, gts = [], []
    for t in range(args.frame_index, end):
        sample = ds[t]
        batch = preprocessor(to_batch(sample, args.device))
        with torch.no_grad():
            action_norm = policy.select_action(batch)  # (B, action_dim), normalized
        action = postprocessor(action_norm.squeeze(0)).cpu()
        preds.append(action)
        gts.append(sample["action"])

    pred = torch.stack(preds)
    gt = torch.stack(gts)
    err = (pred - gt).abs()
    print(f"\npred : shape={tuple(pred.shape)}  mean|err|={err.mean():.4f}  "
          f"max|err|={err.max():.4f}")

    n = min(args.show_first_n, gt.shape[0])
    print(f"\nfirst {n} action steps  (left: GT, right: pred)")
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    for t in range(n):
        print(f"t={t:3d}  gt={gt[t].numpy()}")
        print(f"       pr={pred[t].numpy()}")


if __name__ == "__main__":
    main()
