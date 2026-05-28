"""Convert Isaac Lab teleop HDF5 recordings (remove_hook hookonly) to LeRobot format.

Source layout (per robot variant, e.g. dg5f / rh56f1):
    <src_root>/<SESSION>/
        teleop_recorded_<variant>_<SESSION>.hdf5           # data: demo_i/{actions,obs,processed_actions,initial_state}
        teleop_recorded_<variant>_<SESSION>_images.hdf5    # data: demo_i/{d405_rgb,zivid_rgb}

This script flattens every (SESSION, demo_i) pair into a LeRobot episode, keeping only
the d405 camera (resized) and the full obs vector as observation.state.
"""

import argparse
import logging
from pathlib import Path

import cv2
import h5py
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


ROBOT_SPECS = {
    "dg5f": {
        "action_dim": 26,
        "state_dim": 163,
        "robot_type": "HDR35_20+DG5F_L",
        "main_glob": "teleop_recorded_dg5f_hookonly_*.hdf5",
        "images_suffix": "_images.hdf5",
    },
    "rh56f1": {
        "action_dim": 12,
        "state_dim": 141,
        "robot_type": "HDR35_20+RH56F1_R",
        "main_glob": "teleop_recorded_rh56f1_hookonly_*.hdf5",
        "images_suffix": "_images.hdf5",
    },
}

FPS = 60
TASK_NAME = "remove hook ring from chassis"


CAMERA_HDF5_KEYS = {
    "d405": "d405_rgb",
    "zivid": "zivid_rgb",
}

# HDF5 keys for per-camera binary segmentation masks (only zivid carries one in the
# current release). Stored as 1-channel boolean tensors so the LeRobot autodetector
# tags them as FeatureType.MASK (see lerobot/datasets/utils.py).
MASK_HDF5_KEYS = {
    "zivid": "zivid_mask",
}


def build_features(
    action_dim: int,
    state_dim: int,
    img_h: int,
    img_w: int,
    cameras: list[str],
    masks: list[str] | None = None,
) -> dict:
    features: dict = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": {"axes": [f"s{i}" for i in range(state_dim)]},
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": {"axes": [f"a{i}" for i in range(action_dim)]},
        },
    }
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (img_h, img_w, 3),
            "names": ["height", "width", "channels"],
        }
    for cam in masks or []:
        # Pack the binary mask into uint8 via np.packbits to avoid the ~50x slower
        # pyarrow nested-list decode path for 3D bool tensors.
        # The unpacked shape (img_h × img_w × 1) is encoded in `names` so the model
        # side can reshape after unpacking. `lerobot.datasets.utils.get_features_from_robot`
        # picks up `observation.masks.*` as FeatureType.MASK regardless of shape.
        packed_len = (img_h * img_w + 7) // 8
        features[f"observation.masks.{cam}"] = {
            "dtype": "uint8",
            "shape": (packed_len,),
            "names": ["packed_bits"],
        }
    return features


def iter_sessions(src_root: Path, main_glob: str):
    """Yield (session_dir, main_hdf5_path, images_hdf5_path) tuples.

    Supports two layouts:
      <src>/<SESSION>/teleop_recorded_*.hdf5                    (direct)
      <src>/<COLLECTION>/<SESSION>/teleop_recorded_*.hdf5       (nested one level)
    A given source root may mix both — e.g. dg5f ships a single
    260429_152811/ session next to a dg5f_39traj_v1/ collection wrapping nine
    sessions. We walk every sub-tree and pick directories that contain a
    matching main hdf5 (and exclude the *_images.hdf5 sibling).
    """
    seen: set[Path] = set()
    for candidate in sorted(src_root.rglob(main_glob)):
        if candidate.name.endswith("_images.hdf5"):
            continue
        if candidate.parent in seen:
            # Two main hdf5s in the same directory is unexpected.
            raise RuntimeError(
                f"multiple main hdf5 in {candidate.parent}: already saw a sibling"
            )
        session_dir = candidate.parent
        seen.add(session_dir)
        images_path = candidate.with_name(candidate.stem + "_images.hdf5")
        if not images_path.exists():
            logging.warning("skipping %s: missing %s", session_dir, images_path.name)
            continue
        yield session_dir, candidate, images_path


def convert(
    src_root: Path,
    out_root: Path,
    robot: str,
    img_h: int,
    img_w: int,
    action_field: str,
    cameras: list[str],
    repo_id: str,
    vcodec: str = "libsvtav1",
    streaming_encoding: bool = False,
    batch_encoding_size: int = 1,
    masks: list[str] | None = None,
):
    masks = masks or []
    spec = ROBOT_SPECS[robot]
    features = build_features(
        spec["action_dim"], spec["state_dim"], img_h, img_w, cameras, masks
    )

    create_kwargs = dict(
        repo_id=repo_id,
        fps=FPS,
        features=features,
        root=out_root,
        robot_type=spec["robot_type"],
        use_videos=True,
        vcodec=vcodec,
        streaming_encoding=streaming_encoding,
        batch_encoding_size=batch_encoding_size,
    )
    # When streaming, we feed frames directly into the encoder — no PNG buffer needed.
    if not streaming_encoding:
        create_kwargs["image_writer_processes"] = 0
        create_kwargs["image_writer_threads"] = 4
    dataset = LeRobotDataset.create(**create_kwargs)

    ep_index = 0
    for session_dir, main_path, images_path in iter_sessions(src_root, spec["main_glob"]):
        with h5py.File(main_path, "r") as fmain, h5py.File(images_path, "r") as fimg:
            demo_keys = sorted(fmain["data"].keys(), key=lambda k: int(k.split("_")[1]))
            for demo_key in demo_keys:
                demo = fmain["data"][demo_key]
                if demo_key not in fimg:
                    logging.warning("%s %s: no matching images group, skipped", session_dir.name, demo_key)
                    continue
                img_demo = fimg[demo_key]
                missing_cams = [c for c in cameras if CAMERA_HDF5_KEYS[c] not in img_demo]
                missing_masks = [c for c in masks if MASK_HDF5_KEYS[c] not in img_demo]
                if missing_cams or missing_masks:
                    logging.warning(
                        "%s %s: missing groups cams=%s masks=%s, skipped",
                        session_dir.name, demo_key, missing_cams, missing_masks,
                    )
                    continue

                obs = np.asarray(demo["obs"], dtype=np.float32)
                actions = np.asarray(demo[action_field], dtype=np.float32)
                # Bulk-load entire camera arrays into RAM in one HDF5 read.
                # Individual h5py slicing is ~7x slower than a single contiguous read,
                # and the dataset is uncompressed uint8 so memory cost is bounded.
                cam_arrays = {
                    c: np.asarray(img_demo[CAMERA_HDF5_KEYS[c]][:]) for c in cameras
                }
                mask_arrays = {
                    c: np.asarray(img_demo[MASK_HDF5_KEYS[c]][:]) for c in masks
                }

                T_inputs = [obs.shape[0], actions.shape[0]]
                T_inputs += [arr.shape[0] for arr in cam_arrays.values()]
                T_inputs += [arr.shape[0] for arr in mask_arrays.values()]
                T = min(T_inputs)
                if obs.shape[1] != spec["state_dim"] or actions.shape[1] != spec["action_dim"]:
                    raise RuntimeError(
                        f"shape mismatch in {main_path}:{demo_key} "
                        f"obs={obs.shape} action={actions.shape} expected "
                        f"state={spec['state_dim']} action={spec['action_dim']}"
                    )

                logging.info(
                    "episode %d  session=%s  demo=%s  T=%d  cams=%s  masks=%s",
                    ep_index, session_dir.name, demo_key, T, cameras, masks,
                )

                for t in range(T):
                    frame: dict = {
                        "observation.state": obs[t],
                        "action": actions[t],
                        "task": TASK_NAME,
                    }
                    for cam in cameras:
                        raw = cam_arrays[cam][t]  # (H, W, 3) uint8
                        frame[f"observation.images.{cam}"] = cv2.resize(
                            raw, (img_w, img_h), interpolation=cv2.INTER_AREA
                        )
                    for cam in masks:
                        m = mask_arrays[cam][t]  # (H, W) uint8 or bool
                        # NEAREST keeps the mask binary instead of producing intermediate values.
                        m = cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                        # Pack into uint8 — 50x faster parquet decode than 3D bool.
                        # Decoder must np.unpackbits(...) and slice to img_h*img_w.
                        flat = (m > 0).astype(np.uint8).ravel()
                        frame[f"observation.masks.{cam}"] = np.packbits(flat)
                    dataset.add_frame(frame)
                dataset.save_episode()
                ep_index += 1

    logging.info("Wrote %d episodes to %s", ep_index, out_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="root with <SESSION>/*.hdf5 folders")
    parser.add_argument("--out", type=Path, required=True, help="target LeRobot dataset dir")
    parser.add_argument("--robot", choices=list(ROBOT_SPECS), required=True)
    parser.add_argument("--img-h", type=int, default=240)
    parser.add_argument("--img-w", type=int, default=320)
    parser.add_argument("--action-field", choices=["actions", "processed_actions"], default="actions")
    parser.add_argument(
        "--cameras",
        default="d405",
        help="comma-separated subset of {d405,zivid}. e.g. 'd405,zivid' for multi-cam.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="LeRobot repo_id (defaults to local/<robot>_hookonly).",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--vcodec",
        default="libsvtav1",
        help="video codec. 'auto' picks the best HW encoder (e.g. h264_nvenc on NVIDIA).",
    )
    parser.add_argument(
        "--streaming-encoding",
        action="store_true",
        help="encode frames in-flight (no PNG round-trip) — large speedup, esp. with HW encoders.",
    )
    parser.add_argument(
        "--batch-encoding-size",
        type=int,
        default=1,
        help="number of episodes to buffer before encoding (>1 amortizes encoder startup).",
    )
    parser.add_argument(
        "--masks",
        default="",
        help="comma-separated subset of {zivid} to include binary masks. "
             "Stored as observation.masks.<cam> with dtype bool (H, W, 1).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    unknown = [c for c in cameras if c not in CAMERA_HDF5_KEYS]
    if unknown:
        raise SystemExit(f"unknown camera(s) {unknown}; supported: {sorted(CAMERA_HDF5_KEYS)}")
    masks = [c.strip() for c in args.masks.split(",") if c.strip()]
    unknown_masks = [c for c in masks if c not in MASK_HDF5_KEYS]
    if unknown_masks:
        raise SystemExit(f"unknown mask cam(s) {unknown_masks}; supported: {sorted(MASK_HDF5_KEYS)}")
    repo_id = args.repo_id or f"local/{args.robot}_hookonly"
    convert(
        args.src, args.out, args.robot, args.img_h, args.img_w,
        args.action_field, cameras, repo_id,
        vcodec=args.vcodec,
        streaming_encoding=args.streaming_encoding,
        batch_encoding_size=args.batch_encoding_size,
        masks=masks,
    )


if __name__ == "__main__":
    main()
