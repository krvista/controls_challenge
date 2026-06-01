"""Head-to-head validation on the held-out 1000-seg set (or any range)."""
import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import evalutil  # noqa: E402


def run(name, controller, idxs, params_file=None):
    if params_file and Path(params_file).exists():
        os.environ["FF_PID_PARAMS"] = Path(params_file).read_text().strip()
    else:
        os.environ.pop("FF_PID_PARAMS", None)
    r = evalutil.evaluate(controller, idxs, workers=16, chunksize=4)
    print(f"{name:<16} total={r['total_cost']:7.3f}  lat={r['lataccel_cost']:6.3f}  "
          f"jerk={r['jerk_cost']:6.3f}  median={r['total_median']:6.2f}  p95={r['total_p95']:7.2f}  n={r['n']}")
    return r


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--start", type=int, default=3000)
    args = ap.parse_args()
    idxs = list(range(args.start, args.start + args.n))
    print(f"Validation on segs [{args.start},{args.start+args.n}):\n")
    run("pid", "pid", idxs)
    run("ff_pid(v1-best)", "ff_pid", idxs, ROOT / "controllers" / "ff_pid_best.json")
    run("ff_pid3(v3-best)", "ff_pid3", idxs, ROOT / "controllers" / "ff_pid3_best.json")
