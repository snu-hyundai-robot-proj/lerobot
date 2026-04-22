"""Run a trained ACT checkpoint on a single frame of the converted hookonly dataset.

Loads the policy + its pre/post-processors from the saved checkpoint, picks one frame
from the LeRobot dataset, runs `predict_action_chunk`, and compares the predicted
chunk against the ground-truth actions that follow that frame.

Example
-------
python examples/port_datasets/inference_act_hookonly.py \
    --checkpoint outputs/train/dg5f_act/checkpoints/last/pretrained_model \
    --dataset-root /home/ngseo/recorded_dataset/lerobot/dg5f_hookonly \
    --frame-index 0
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def load_policy(checkpoint: Path, device: str):
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device = device
    policy = ACTPolicy.from_pretrained(checkpoint, config=cfg).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(cfg, pretrained_path=str(checkpoint))
    return cfg, policy, preprocessor, postprocessor


def get_frame_with_future_actions(ds: LeRobotDataset, frame_index: int, horizon: int) -> tuple[dict, torch.Tensor]:
    """Return (single-frame dict, ground-truth action chunk of length `horizon`)."""
    sample = ds[frame_index]
    ep_idx = int(sample["episode_index"])
    ep_meta = ds.meta.episodes[ep_idx]
    ep_start = int(ep_meta["dataset_from_index"])
    ep_end = int(ep_meta["dataset_to_index"])  # exclusive
    chunk_end = min(frame_index + horizon, ep_end)
    gt = torch.stack([ds[i]["action"] for i in range(frame_index, chunk_end)])
    pad = horizon - gt.shape[0]
    if pad > 0:
        gt = torch.cat([gt, gt[-1:].expand(pad, -1)], dim=0)
    return sample, gt, ep_idx, ep_start, ep_end


def to_batch(sample: dict, device: str) -> dict:
    """Add batch dim to tensor entries; keep `task` as a list (preprocessor expects it)."""
    batch = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(device)
        elif isinstance(v, str):
            batch[k] = [v]
        else:
            batch[k] = v
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="path to .../checkpoints/<step>/pretrained_model")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default=None,
                        help="dataset repo id (defaults to the dataset-root folder name under local/)")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--show-first-n", type=int, default=10,
                        help="number of action steps to print for comparison")
    args = parser.parse_args()

    repo_id = args.repo_id or f"local/{args.dataset_root.name}"
    ds = LeRobotDataset(repo_id, root=args.dataset_root)

    cfg, policy, preprocessor, postprocessor = load_policy(args.checkpoint, args.device)
    horizon = cfg.chunk_size
    print(f"loaded ACT  chunk_size={horizon}  device={args.device}  ckpt={args.checkpoint}")

    sample, gt_chunk, ep_idx, ep_start, ep_end = get_frame_with_future_actions(
        ds, args.frame_index, horizon
    )
    print(f"frame {args.frame_index}  episode {ep_idx} ({ep_start}..{ep_end})  "
          f"obs.state={tuple(sample['observation.state'].shape)}  "
          f"img={tuple(sample['observation.images.d405'].shape)}")

    batch = to_batch(sample, args.device)
    batch = preprocessor(batch)

    with torch.no_grad():
        pred_norm = policy.predict_action_chunk(batch)  # (B, horizon, action_dim) - normalized space

    pred = postprocessor(pred_norm.squeeze(0)).cpu()  # (horizon, action_dim) in original units
    gt = gt_chunk.cpu()

    err = (pred - gt).abs()
    print(f"\npred chunk : shape={tuple(pred.shape)}  mean|err|={err.mean():.4f}  "
          f"max|err|={err.max():.4f}")

    n = min(args.show_first_n, horizon, gt.shape[0])
    print(f"\nfirst {n} action steps  (left: GT, right: pred)")
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    for t in range(n):
        print(f"t={t:3d}  gt={gt[t].numpy()}")
        print(f"       pr={pred[t].numpy()}")


if __name__ == "__main__":
    main()
