"""CMA-ES tuner for the ff_pid controller.

Each candidate parameter vector is written to FF_PID_PARAMS and evaluated as the
mean total_cost over a fixed tune subset (parallel rollouts). cma works in a
normalized [0,1] box; we scale to per-parameter ranges. lookahead is rounded to int.
"""
import os
import sys
import json
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import cma  # noqa: E402
import evalutil  # noqa: E402

# name -> (lo, hi, is_int)
SPACE = [
    ("ff_gain",   0.6, 1.4, False),
    ("lookahead", 0.0, 8.0, True),
    ("kp",        0.0, 0.5, False),
    ("ki",        0.0, 0.2, False),
    ("kd",       -0.2, 0.1, False),
    ("i_clip",    0.1, 3.0, False),
    ("ref_ema",   0.0, 0.7, False),
]


def decode(x):
    p = {}
    for xi, (name, lo, hi, is_int) in zip(x, SPACE):
        v = lo + np.clip(xi, 0, 1) * (hi - lo)
        p[name] = int(round(v)) if is_int else float(v)
    return p


def fitness(x, idxs):
    p = decode(x)
    os.environ["FF_PID_PARAMS"] = json.dumps(p)
    res = evalutil.evaluate("ff_pid", idxs, workers=16, chunksize=2)
    return res["total_cost"]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_tune", type=int, default=80, help="segments per fitness eval")
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--x0", default="", help="optional JSON params to seed from")
    args = ap.parse_args()

    tune_idxs = evalutil.TUNE_IDXS[:args.n_tune]

    if args.x0:
        seed_p = json.loads(args.x0)
        x0 = []
        for name, lo, hi, _ in SPACE:
            x0.append((seed_p.get(name, (lo + hi) / 2) - lo) / (hi - lo))
        x0 = np.clip(x0, 0, 1)
    else:
        x0 = [0.5] * len(SPACE)

    es = cma.CMAEvolutionStrategy(
        list(x0), args.sigma,
        {"popsize": args.popsize, "bounds": [0, 1], "seed": args.seed, "maxiter": args.gens},
    )

    best_f, best_p = float("inf"), None
    gen = 0
    t0 = time.time()
    while not es.stop():
        sols = es.ask()
        fits = [fitness(x, tune_idxs) for x in sols]
        es.tell(sols, fits)
        gen += 1
        gi = int(np.argmin(fits))
        if fits[gi] < best_f:
            best_f, best_p = fits[gi], decode(sols[gi])
        print(f"[gen {gen:>2d}] best_gen={min(fits):.3f} best_all={best_f:.3f} "
              f"elapsed={time.time()-t0:.0f}s params={json.dumps(best_p)}", flush=True)

    print("\n=== BEST ===")
    print(f"total_cost (tune n={args.n_tune}) = {best_f:.4f}")
    print(json.dumps(best_p))
    with open(ROOT / "controllers" / "ff_pid_best.json", "w") as f:
        json.dump(best_p, f, indent=2)
    print(f"saved -> controllers/ff_pid_best.json")
