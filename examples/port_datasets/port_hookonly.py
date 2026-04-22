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


def build_features(action_dim: int, state_dim: int, img_h: int, img_w: int) -> dict:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": {"axes": [f"s{i}" for i in range(state_dim)]},
        },
        "observation.images.d405": {
            "dtype": "video",
            "shape": (img_h, img_w, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": {"axes": [f"a{i}" for i in range(action_dim)]},
        },
    }


def iter_sessions(src_root: Path, main_glob: str):
    """Yield (session_dir, main_hdf5_path, images_hdf5_path) tuples."""
    for session_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        main = [p for p in session_dir.glob(main_glob) if not p.name.endswith("_images.hdf5")]
        if not main:
            logging.warning("skipping %s: no main hdf5 matching %s", session_dir, main_glob)
            continue
        if len(main) > 1:
            raise RuntimeError(f"multiple main hdf5 in {session_dir}: {main}")
        main_path = main[0]
        images_path = main_path.with_name(main_path.stem + "_images.hdf5")
        if not images_path.exists():
            logging.warning("skipping %s: missing %s", session_dir, images_path.name)
            continue
        yield session_dir, main_path, images_path


def convert(
    src_root: Path,
    out_root: Path,
    robot: str,
    img_h: int,
    img_w: int,
    action_field: str,
):
    spec = ROBOT_SPECS[robot]
    features = build_features(spec["action_dim"], spec["state_dim"], img_h, img_w)

    dataset = LeRobotDataset.create(
        repo_id=f"local/{robot}_hookonly",
        fps=FPS,
        features=features,
        root=out_root,
        robot_type=spec["robot_type"],
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4,
    )

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

                obs = np.asarray(demo["obs"], dtype=np.float32)
                actions = np.asarray(demo[action_field], dtype=np.float32)
                d405 = img_demo["d405_rgb"]  # (T, H, W, 3) uint8

                T = min(obs.shape[0], actions.shape[0], d405.shape[0])
                if obs.shape[1] != spec["state_dim"] or actions.shape[1] != spec["action_dim"]:
                    raise RuntimeError(
                        f"shape mismatch in {main_path}:{demo_key} "
                        f"obs={obs.shape} action={actions.shape} expected "
                        f"state={spec['state_dim']} action={spec['action_dim']}"
                    )

                logging.info(
                    "episode %d  session=%s  demo=%s  T=%d", ep_index, session_dir.name, demo_key, T
                )

                for t in range(T):
                    raw_img = d405[t]  # (H, W, 3) uint8
                    img = cv2.resize(raw_img, (img_w, img_h), interpolation=cv2.INTER_AREA)
                    dataset.add_frame(
                        {
                            "observation.state": obs[t],
                            "observation.images.d405": img,
                            "action": actions[t],
                            "task": TASK_NAME,
                        }
                    )
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
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    convert(args.src, args.out, args.robot, args.img_h, args.img_w, args.action_field)


if __name__ == "__main__":
    main()
