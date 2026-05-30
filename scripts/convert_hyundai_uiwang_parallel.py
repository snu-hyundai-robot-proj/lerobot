#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parallel driver for the Hyundai Uiwang -> LeRobot conversion.

The single-process converter only saturates ~1 of the 20 cores (the main loop
does decode + resize + synchronous PNG writes). This driver shards the episodes
across ``--workers`` subprocesses, each producing an independent LeRobot dataset,
then merges them with ``aggregate_datasets`` (which copies/reindexes videos+data
without re-encoding).

Usage:
    python scripts/convert_hyundai_uiwang_parallel.py --side left  --workers 10
    python scripts/convert_hyundai_uiwang_parallel.py --side right --workers 10
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
RAW_ROOT = Path.home() / "hyundai_uiwang_data"
CONVERTER = REPO_ROOT / "scripts" / "convert_hyundai_uiwang_to_lerobot.py"


def episode_indices(side: str) -> list[int]:
    sysdir = RAW_ROOT / side / "datas" / "system"
    idxs = []
    for p in sysdir.glob("frame_data_*.bin"):
        try:
            idxs.append(int(p.stem.split("_")[-1]))
        except ValueError:
            continue
    return sorted(idxs)


def shard(items: list[int], k: int) -> list[list[int]]:
    """Split into k contiguous near-equal groups."""
    n = len(items)
    k = min(k, n)
    base, extra = divmod(n, k)
    out, start = [], 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        out.append(items[start : start + size])
        start += size
    return [g for g in out if g]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["left", "right"], required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--vcodec", default="h264")
    ap.add_argument("--limit", type=int, default=None, help="only first N episodes (debug)")
    ap.add_argument("--keep-shards", action="store_true", help="don't delete shard datasets after merge")
    args = ap.parse_args()

    eps = episode_indices(args.side)
    if args.limit:
        eps = eps[: args.limit]
    groups = shard(eps, args.workers)
    print(f"[parallel] side={args.side} episodes={len(eps)} workers={len(groups)}")

    shards_dir = REPO_ROOT / "data" / "lerobot" / "_shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    final_root = REPO_ROOT / "data" / "lerobot" / f"hyundai_uiwang_{args.side}"
    if final_root.exists():
        shutil.rmtree(final_root)

    # --- launch worker subprocesses ------------------------------------------
    procs = []
    shard_roots, shard_repo_ids = [], []
    for i, g in enumerate(groups):
        root = shards_dir / f"{args.side}_shard{i:02d}"
        repo_id = f"local/hyundai_uiwang_{args.side}_shard{i:02d}"
        shard_roots.append(root)
        shard_repo_ids.append(repo_id)
        logf = open(shards_dir / f"{args.side}_shard{i:02d}.log", "w")
        cmd = [
            sys.executable, str(CONVERTER),
            "--side", args.side,
            "--episodes", ",".join(map(str, g)),
            "--root", str(root),
            "--repo-id", repo_id,
            "--vcodec", args.vcodec,
            "--overwrite",
        ]
        env = {"PYTHONPATH": str(REPO_ROOT / "src"), "OMP_NUM_THREADS": "1",
               "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        import os
        full_env = {**os.environ, **env}
        procs.append((i, subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=full_env), logf))
        print(f"  worker {i:02d}: {len(g)} episodes (bins {g[0]}..{g[-1]}) -> {root.name}")

    # --- wait ----------------------------------------------------------------
    t0 = time.time()
    failed = []
    for i, p, logf in procs:
        rc = p.wait()
        logf.close()
        status = "ok" if rc == 0 else f"FAILED(rc={rc})"
        print(f"  worker {i:02d} done: {status}  ({time.time() - t0:.0f}s elapsed)")
        if rc != 0:
            failed.append(i)
    if failed:
        raise SystemExit(f"workers failed: {failed} — see {shards_dir}/{args.side}_shard*.log")

    # --- merge ---------------------------------------------------------------
    print("[parallel] aggregating shards ...")
    from lerobot.datasets.aggregate import aggregate_datasets

    aggregate_datasets(
        repo_ids=shard_repo_ids,
        aggr_repo_id=f"local/hyundai_uiwang_{args.side}",
        roots=shard_roots,
        aggr_root=final_root,
    )
    print(f"[parallel] merged -> {final_root}")

    if not args.keep_shards:
        for r in shard_roots:
            shutil.rmtree(r, ignore_errors=True)
        print("[parallel] shard datasets removed")

    print(f"[parallel] DONE side={args.side} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
