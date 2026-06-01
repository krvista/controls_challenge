"""Reusable evaluation harness for controller development/tuning.

Headless (no plotting), deterministic per-segment (seed fixed by tinyphysics),
and lets us pick explicit segment subsets so tuning vs. validation never overlap.
"""
import os
import sys
import numpy as np
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tinyphysics import run_rollout  # noqa: E402

MODEL_PATH = str(ROOT / "models" / "tinyphysics.onnx")
DATA_DIR = ROOT / "data"


def seg_files(idxs):
    """List of CSV paths for the given integer segment indices."""
    return [DATA_DIR / f"{i:05d}.csv" for i in idxs]


def _one(data_path, controller_type, model_path):
    cost, _, _ = run_rollout(data_path, controller_type, model_path, debug=False)
    return cost


def evaluate(controller_type, idxs, model_path=MODEL_PATH, workers=16, chunksize=4):
    """Run controller over the given segment indices, return dict of mean costs + arrays."""
    from tqdm.contrib.concurrent import process_map
    files = [str(f) for f in seg_files(idxs)]
    fn = partial(_one, controller_type=controller_type, model_path=model_path)
    results = process_map(fn, files, max_workers=workers, chunksize=chunksize)
    lat = np.array([r["lataccel_cost"] for r in results])
    jerk = np.array([r["jerk_cost"] for r in results])
    tot = np.array([r["total_cost"] for r in results])
    return {
        "lataccel_cost": float(lat.mean()),
        "jerk_cost": float(jerk.mean()),
        "total_cost": float(tot.mean()),
        "total_median": float(np.median(tot)),
        "total_p95": float(np.percentile(tot, 95)),
        "n": len(tot),
        "_lat": lat, "_jerk": jerk, "_tot": tot,
    }


# Fixed splits (never overlap): keep tuning honest.
TUNE_IDXS = list(range(0, 300))
VAL_IDXS = list(range(3000, 4000))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--controller", default="pid")
    p.add_argument("--split", default="tune", choices=["tune", "val", "custom"])
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    if args.split == "tune":
        idxs = TUNE_IDXS
    elif args.split == "val":
        idxs = VAL_IDXS
    else:
        idxs = list(range(args.start, args.start + args.n))

    res = evaluate(args.controller, idxs, workers=args.workers)
    print(f"\ncontroller={args.controller} split={args.split} n={res['n']}")
    print(f"  lataccel_cost = {res['lataccel_cost']:.4f}")
    print(f"  jerk_cost     = {res['jerk_cost']:.4f}")
    print(f"  total_cost    = {res['total_cost']:.4f}  (median {res['total_median']:.2f}, p95 {res['total_p95']:.2f})")
