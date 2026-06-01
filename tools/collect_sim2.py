"""On-policy sim system-ID with EXPLORATION (action dither) + stable reparameterization.

Runs the sim with ff_pid3 (good policy) plus small random dither added to the action,
so the realized lataccel delta decorrelates from the operating point. Logs:
    (v_ego, roll, a_ego, current L_t, applied action u_t, next L_{t+1})

Fit the well-conditioned inverse, per v-bin:
    u = [a1*L_t + a2*roll + a3*L_t|L_t| + a4*a_ego + a5]            (HOLD steer)
        + k * (L_{t+1} - L_t)                                       (TRANSIENT gain)
The dither gives (L_{t+1}-L_t) independent variance, so k is identifiable and the
hold block is decoupled. Controller FF: u = hold(current,...) + k*(ref - current).
Saved as ff_model_v5.npz with arrays v_centers, hold (n,5), kgain (n,).
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


class DitherLogController(FF3):
    def __init__(self, sink, dither, rng):
        super().__init__()
        self.sink = sink
        self.dither = dither
        self.rng = rng
        self._prev = None

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        if self._prev is not None and self.step >= 100 - 20:
            v, roll, a, Lt, ut = self._prev
            self.sink.append((v, roll, a, Lt, ut, current_lataccel))
        u = super().update(target_lataccel, current_lataccel, state, future_plan)
        u = float(np.clip(u + self.rng.normal(0, self.dither), -2, 2))
        self._prev = (state.v_ego, state.roll_lataccel, state.a_ego, current_lataccel, u)
        return u  # dithered action is what the sim applies


def collect(idxs, params, dither):
    os.environ["FF_PID_PARAMS"] = json.dumps(params)
    model = tp.TinyPhysicsModel(str(ROOT / "models" / "tinyphysics.onnx"), debug=False)
    sink = []
    for i in idxs:
        rng = np.random.default_rng(i)
        ctrl = DitherLogController(sink, dither, rng)
        sim = tp.TinyPhysicsSimulator(model, str(ROOT / "data" / f"{i:05d}.csv"), controller=ctrl, debug=False)
        sim.rollout()
    return np.array(sink)


def fit(data, out):
    v, roll, a, Lt, ut, Ln = data.T
    finite = np.all(np.isfinite(data), axis=1)
    d = Ln - Lt
    centers, hold, kg = [], [], []
    for lo, hi in zip(V_BINS[:-1], V_BINS[1:]):
        m = (v >= lo) & (v < hi) & finite
        if m.sum() < 200:
            continue
        X = np.column_stack([Lt[m], roll[m], Lt[m] * np.abs(Lt[m]), a[m], np.ones(m.sum()), d[m]])
        coef, *_ = np.linalg.lstsq(X, ut[m], rcond=None)
        r2 = 1 - np.sum((ut[m] - X @ coef) ** 2) / np.sum((ut[m] - ut[m].mean()) ** 2)
        centers.append((lo + min(hi, 45)) / 2)
        hold.append(coef[:5]); kg.append(coef[5])
        print(f"  v[{lo:.0f},{hi:.0f}) n={m.sum():>6d}  k(trans)={coef[5]:+.3f}  "
              f"hold[L={coef[0]:+.3f} roll={coef[1]:+.3f} sat={coef[2]:+.4f} c={coef[4]:+.3f}]  R2={r2:.3f}")
    np.savez(out, v_centers=np.array(centers), hold=np.array(hold), kgain=np.array(kg))
    print(f"saved -> {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--spread", type=int, default=11)
    ap.add_argument("--dither", type=float, default=0.15)
    ap.add_argument("--params", default=str(ROOT / "controllers" / "ff_pid3_best.json"))
    ap.add_argument("--out", default=str(ROOT / "controllers" / "ff_model_v5.npz"))
    args = ap.parse_args()
    params = json.loads(Path(args.params).read_text()) if Path(args.params).exists() else {}
    idxs = list(range(0, args.n * args.spread, args.spread))
    print(f"Collecting (dither={args.dither}) over {len(idxs)} segs...")
    data = collect(idxs, params, args.dither)
    print(f"transitions: {len(data)}")
    fit(data, args.out)
