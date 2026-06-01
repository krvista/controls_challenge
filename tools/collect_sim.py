"""On-policy collection of the SIM's inverse dynamics.

Runs the simulator (single process) with a logging controller wrapping the current
best ff_pid3, recording per control-step tuples:
    (v_ego, roll, a_ego, current_lataccel L_t, action u_t, next_lataccel L_{t+1})
We then fit the sim-accurate one-step inverse:
    steer = g(target=L_{t+1}, current=L_t, roll, v_ego, a_ego)
which the controller uses as feedforward. This corrects the sim-vs-realdata mismatch
that limits tracking accuracy.
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import tinyphysics as tp  # noqa: E402
from controllers.ff_pid3 import Controller as FF3  # noqa: E402

V_BINS = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 100], dtype=float)


class LogController(FF3):
    """Wraps ff_pid3, logging (state, action, lataccel) transitions."""
    def __init__(self, sink):
        super().__init__()
        self.sink = sink
        self._prev = None  # (v, roll, a, L_t, u_t)

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # current_lataccel here is the result of the previous action (L_{t})
        if self._prev is not None and self.step >= 100 - 20:
            v, roll, a, Lt, ut = self._prev
            self.sink.append((v, roll, a, Lt, ut, current_lataccel))
        u = super().update(target_lataccel, current_lataccel, state, future_plan)
        self._prev = (state.v_ego, state.roll_lataccel, state.a_ego, current_lataccel, u)
        return u


def collect(idxs, params):
    os.environ["FF_PID_PARAMS"] = json.dumps(params)
    model = tp.TinyPhysicsModel(str(ROOT / "models" / "tinyphysics.onnx"), debug=False)
    sink = []
    for i in idxs:
        path = str(ROOT / "data" / f"{i:05d}.csv")
        ctrl = LogController(sink)
        sim = tp.TinyPhysicsSimulator(model, path, controller=ctrl, debug=False)
        sim.rollout()
    return np.array(sink)  # cols: v, roll, a, Lt, ut, Lnext


def fit(data, out):
    v, roll, a, Lt, ut, Ln = data.T
    finite = np.all(np.isfinite(data), axis=1)
    centers, coefs = [], []
    for lo, hi in zip(V_BINS[:-1], V_BINS[1:]):
        m = (v >= lo) & (v < hi) & finite
        if m.sum() < 200:
            continue
        # features: [target, current, roll, target*|target|, a, 1]
        X = np.column_stack([Ln[m], Lt[m], roll[m], Ln[m] * np.abs(Ln[m]), a[m], np.ones(m.sum())])
        coef, *_ = np.linalg.lstsq(X, ut[m], rcond=None)
        pred = X @ coef
        r2 = 1 - np.sum((ut[m] - pred) ** 2) / np.sum((ut[m] - ut[m].mean()) ** 2)
        centers.append((lo + min(hi, 45)) / 2)
        coefs.append(coef)
        print(f"  v[{lo:.0f},{hi:.0f}) n={m.sum():>6d}  tgt={coef[0]:+.3f} cur={coef[1]:+.3f} "
              f"roll={coef[2]:+.3f} sat={coef[3]:+.4f} a={coef[4]:+.4f} c={coef[5]:+.4f}  R2={r2:.3f}")
    np.savez(out, v_centers=np.array(centers), coefs=np.array(coefs))
    print(f"saved -> {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--spread", type=int, default=8, help="segment stride for representativeness")
    ap.add_argument("--params", default=str(ROOT / "controllers" / "ff_pid3_best.json"))
    ap.add_argument("--out", default=str(ROOT / "controllers" / "ff_model_v3.npz"))
    args = ap.parse_args()
    params = json.loads(Path(args.params).read_text()) if Path(args.params).exists() else {}
    idxs = list(range(0, args.n * args.spread, args.spread))
    print(f"Collecting sim transitions over {len(idxs)} segments (stride {args.spread})...")
    data = collect(idxs, params)
    print(f"transitions: {len(data)}")
    print("Fitting sim-accurate one-step inverse:")
    fit(data, args.out)
