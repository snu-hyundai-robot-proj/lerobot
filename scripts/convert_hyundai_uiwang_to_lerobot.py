#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert Hyundai Uiwang raw captures into a proper LeRobot v3.0 dataset.

Unlike the team's ``convert_data_aligned_video.py`` (which only emits parquet +
info.json and never places the videos or the v3.0 metadata), this script drives
LeRobot's official dataset API (``LeRobotDataset.create`` / ``add_frame`` /
``save_episode``). That guarantees:

  * videos encoded to h264 in the correct ``videos/<key>/chunk-*/file-*.mp4`` layout
  * complete metadata (meta/episodes, meta/tasks.parquet, meta/stats.json, info.json)
  * a dataset that actually loads for training

The bin parsing and the video-timeline resampling are reused verbatim from the
team's converter so the per-frame state/action alignment is identical.

Camera mapping (both resized to 640x480):
    observation.images.front_rgb  <-  {side}_{i}_zivid_video.mp4
    observation.images.wrist_rgb  <-  {side}_{i}_wrist_cam_video.mp4

Feature mapping (per frame, matching the team's build_rows):
    observation.state          = robot_joint(6)        + gripper_joint(20)        -> 26
    action                     = target_robot_joint(6) + target_gripper_joint(20) -> 26
    observation.gripper_sensor = gripper_sensor(30)
    observation.wrist_ft_sensor= robot_ft(6)

Usage:
    python scripts/convert_hyundai_uiwang_to_lerobot.py --side left --limit 1   # smoke
    python scripts/convert_hyundai_uiwang_to_lerobot.py --side left             # all episodes
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# --- reuse the team's bin parser + video-timeline resampler verbatim ----------
RAW_ROOT = Path.home() / "hyundai_uiwang_data"
_TEAM_CONVERTER = RAW_ROOT / "convert_data_aligned_video.py"

_spec = importlib.util.spec_from_file_location("team_converter", _TEAM_CONVERTER)
team = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(team)  # type: ignore[union-attr]

# repo src import (lerobot is a source checkout here, not pip-installed)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

TARGET_W, TARGET_H = 640, 480  # both cameras resized to this (matches info.json shape)
TASK = "hyundai uiwang manipulation"


def build_features() -> dict:
    return {
        "observation.images.front_rgb": {
            "dtype": "video",
            "shape": (TARGET_H, TARGET_W, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist_rgb": {
            "dtype": "video",
            "shape": (TARGET_H, TARGET_W, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (26,),
            "names": {"arm": [f"joint_{i}" for i in range(6)], "hand": [f"joint_{i}" for i in range(20)]},
        },
        "observation.gripper_sensor": {"dtype": "float32", "shape": (30,), "names": None},
        "observation.wrist_ft_sensor": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["fx", "fy", "fz", "tx", "ty", "tz"],
        },
        "action": {
            "dtype": "float32",
            "shape": (26,),
            "names": {"arm": [f"joint_{i}" for i in range(6)], "hand": [f"joint_{i}" for i in range(20)]},
        },
    }


def episode_indices(side: str) -> list[int]:
    sysdir = RAW_ROOT / side / "datas" / "system"
    idxs = []
    for p in sysdir.glob("frame_data_*.bin"):
        try:
            idxs.append(int(p.stem.split("_")[-1]))
        except ValueError:
            continue
    return sorted(idxs)


def resize_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR (from cv2) -> resized RGB uint8 HWC."""
    resized = cv2.resize(frame_bgr, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def add_episode(ds: LeRobotDataset, side: str, ep: int) -> int:
    base = RAW_ROOT / side / "datas"
    bin_path = base / "system" / f"frame_data_{ep}.bin"
    wrist_path = base / "videos" / f"{side}_{ep}_wrist_cam_video.mp4"
    zivid_path = base / "videos" / f"{side}_{ep}_zivid_video.mp4"
    for p in (bin_path, wrist_path, zivid_path):
        if not p.exists():
            raise FileNotFoundError(p)

    records = team.read_state_records(bin_path)
    vinfo = team.inspect_video_info(wrist_path)  # wrist is the timeline reference (matches existing parquet)
    fps = float(vinfo["fps"])
    n_frames = int(vinfo["frame_count"])
    resampled = team.resample_records_by_video_timeline(
        records, video_frame_count=n_frames, video_fps=fps, time_offset_sec=0.0
    )

    cap_w = cv2.VideoCapture(str(wrist_path))
    cap_z = cv2.VideoCapture(str(zivid_path))
    written = 0
    try:
        for i, rec in enumerate(resampled):
            ok_w, fw = cap_w.read()
            ok_z, fz = cap_z.read()
            if not (ok_w and ok_z):
                break  # video ended early; truncate to decodable frames
            state = np.asarray(list(rec["robot_joint"]) + list(rec["gripper_joint"]), dtype=np.float32)
            action = np.asarray(
                list(rec["target_robot_joint"]) + list(rec["target_gripper_joint"]), dtype=np.float32
            )
            ds.add_frame(
                {
                    "observation.images.front_rgb": resize_rgb(fz),
                    "observation.images.wrist_rgb": resize_rgb(fw),
                    "observation.state": state,
                    "observation.gripper_sensor": np.asarray(rec["gripper_sensor"], dtype=np.float32),
                    "observation.wrist_ft_sensor": np.asarray(rec["robot_ft"], dtype=np.float32),
                    "action": action,
                    # timestamp is auto-set by add_frame to frame_index / fps, which equals
                    # i / fps here (uniform video timeline) — so we don't pass it explicitly.
                    "task": TASK,
                }
            )
            written += 1
    finally:
        cap_w.release()
        cap_z.release()

    ds.save_episode()
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["left", "right"], required=True)
    ap.add_argument("--root", default=None, help="output dataset root (default data/lerobot/hyundai_uiwang_<side>)")
    ap.add_argument("--repo-id", default=None, help="dataset repo_id (default local/hyundai_uiwang_<side>)")
    ap.add_argument("--limit", type=int, default=None, help="convert only the first N episodes (smoke test)")
    ap.add_argument(
        "--episodes", default=None, help="comma-separated bin indices to convert (overrides auto-discovery; for sharding)"
    )
    ap.add_argument("--vcodec", default="h264", help="video codec (h264 default; see VALID_VIDEO_CODECS)")
    ap.add_argument("--overwrite", action="store_true", help="delete existing output root first")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_root = Path(args.root) if args.root else repo_root / "data" / "lerobot" / f"hyundai_uiwang_{args.side}"
    repo_id = args.repo_id or f"local/hyundai_uiwang_{args.side}"
    if out_root.exists():
        if args.overwrite:
            shutil.rmtree(out_root)
        else:
            raise SystemExit(f"output root already exists: {out_root} (use --overwrite)")

    if args.episodes:
        eps = [int(x) for x in args.episodes.split(",") if x.strip()]
    else:
        eps = episode_indices(args.side)
    if args.limit:
        eps = eps[: args.limit]
    print(f"[convert] side={args.side} episodes={len(eps)} -> {out_root}")

    # fps is read per-episode from the video; use the first episode's fps for the dataset.
    first_vinfo = team.inspect_video_info(
        RAW_ROOT / args.side / "datas" / "videos" / f"{args.side}_{eps[0]}_wrist_cam_video.mp4"
    )
    fps = int(round(float(first_vinfo["fps"])))

    robot_type = "HDRB + DG-5F-M-LEFT" if args.side == "left" else "HDRB + INSPIRE-RH56"
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=build_features(),
        root=out_root,
        robot_type=robot_type,
        use_videos=True,
        vcodec=args.vcodec,
    )

    total = 0
    for n, ep in enumerate(eps):
        w = add_episode(ds, args.side, ep)
        total += w
        print(f"  [{n + 1}/{len(eps)}] episode bin#{ep}: {w} frames")

    print(f"[done] {len(eps)} episodes, {total} frames -> {out_root}")


if __name__ == "__main__":
    main()
