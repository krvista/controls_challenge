"""CMA-ES tuner for ff_pid3 (forward-window reference + rich inverse model)."""
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

SPACE = [
    ("ff_gain",    0.6, 1.3, False),
    ("ref_center", 0.0, 8.0, False),
    ("ref_sigma",  0.5, 8.0, False),
    ("kp",         0.0, 0.5, False),
    ("ki",         0.0, 0.25, False),
    ("kd",        -0.2, 0.05, False),
    ("i_clip",     0.1, 3.0, False),
]


def decode(x):
    return {name: float(lo + np.clip(xi, 0, 1) * (hi - lo))
            for xi, (name, lo, hi, _) in zip(x, SPACE)}


def fitness(x, idxs):
    os.environ["FF_PID_PARAMS"] = json.dumps(decode(x))
    return evalutil.evaluate("ff_pid3", idxs, workers=16, chunksize=2)["total_cost"]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_tune", type=int, default=100)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--gens", type=int, default=30)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--x0", default="")
    args = ap.parse_args()

    tune_idxs = evalutil.TUNE_IDXS[:args.n_tune]
    if args.x0:
        sp = json.loads(args.x0)
        x0 = np.clip([(sp.get(n, (lo + hi) / 2) - lo) / (hi - lo) for n, lo, hi, _ in SPACE], 0, 1)
    else:
        x0 = [0.5] * len(SPACE)

    es = cma.CMAEvolutionStrategy(list(x0), args.sigma,
        {"popsize": args.popsize, "bounds": [0, 1], "seed": args.seed, "maxiter": args.gens})
    best_f, best_p, gen, t0 = float("inf"), None, 0, time.time()
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

    print(f"\n=== BEST === total_cost(tune n={args.n_tune})={best_f:.4f}\n{json.dumps(best_p)}")
    with open(ROOT / "controllers" / "ff_pid3_best.json", "w") as f:
        json.dump(best_p, f, indent=2)
    print("saved -> controllers/ff_pid3_best.json")
