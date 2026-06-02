from . import BaseController
import json
import os
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Receding-horizon MPC via COORDINATE DESCENT with 1-D golden-section line search
# (no Jacobian -> avoids the GN divergence of mpc3/mpc4). Each control step it
# optimizes the H-step action sequence to minimize the TRUE horizon cost
# (tracking + 2*jerk on raw targets), rolling the verified expected-value plant
# forward on-manifold, then applies the first action and warm-starts the next step.
#
# Designed for correctness/robustness; it is compute-heavy (intended to be run
# offline for the full 5000-seg eval). Tune H / sweeps for the speed-quality
# tradeoff. Predictor verified vs sim (corr 0.9998) by tools/check_model.py.

CONTEXT = 20
VOCAB = 1024
LAT_RANGE = (-5.0, 5.0)
STEER_RANGE = (-2.0, 2.0)
MAX_DELTA = 0.5
TEMP = 0.8
CONTROL_START = 100
FIRST_UPDATE = CONTEXT
_BINS = np.linspace(LAT_RANGE[0], LAT_RANGE[1], VOCAB)
_GR = 0.6180339887498949

DEFAULTS = {
    "H": 10,           # horizon
    "w_jerk": 2.0,     # jerk:tracking ratio (matches true cost)
    "w_du": 1.0,       # action-rate regularizer: keeps actions smooth / ON-MANIFOLD so
                       # the optimizer can't exploit the model with adversarial oscillation
    "sweeps": 2,       # coordinate-descent sweeps per step
    "n_gs": 7,         # golden-section iterations per action
}

_DIR = Path(__file__).resolve().parent


def _load_params():
    p = dict(DEFAULTS)
    env = os.environ.get("FF_PID_PARAMS")
    if env:
        p.update(json.loads(env))
    return p


class Controller(BaseController):
    def __init__(self):
        self.p = _load_params()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        with open(_DIR.parent / "models" / "tinyphysics.onnx", "rb") as f:
            self.sess = ort.InferenceSession(f.read(), opts, ["CPUExecutionProvider"])
        m = np.load(_DIR / "ff_model.npz")
        self.iv, self.ia, self.ic = m["v_centers"], m["a_v"], m["c_v"]
        self.roll_h, self.v_h, self.a_h, self.act_h, self.lat_h = [], [], [], [], []
        self.step = 0
        self.plan = None

    def _seed_action(self, target, roll, v):
        a = np.interp(v, self.iv, self.ia)
        c = np.interp(v, self.iv, self.ic)
        return float(np.clip(a * (target - roll) + c, *STEER_RANGE))

    def _expected1(self, aw, rw, vw, aaw, lw):
        states = np.stack([aw, rw, vw, aaw], axis=1)[None].astype(np.float32)
        toks = np.digitize(np.clip(lw, *LAT_RANGE), _BINS, right=True)[None].astype(np.int64)
        logits = self.sess.run(None, {"states": states, "tokens": toks})[0]
        z = logits[0, -1] / TEMP
        e = np.exp(z - z.max())
        return float((e / e.sum()) @ _BINS)

    def _rollout(self, plan, rolls, vs, as_, H):
        """rolls/vs/as_ : per-horizon-step roll/v/a (index k = step t+k); rolls[0]=roll[t].
        IMPORTANT: roll_h/v_h/a_h already include step t (current), but act_h/lat_h end
        at t-1. So at k=0 we append the action but NOT roll/v/a; for k>=1 append future
        roll/v/a. This matches the verified sim indexing (tools/check_model.py)."""
        al = list(self.act_h); rl = list(self.roll_h)
        vl = list(self.v_h); aal = list(self.a_h); ll = list(self.lat_h)
        out = np.empty(H)
        for k in range(H):
            al.append(plan[k])
            if k > 0:
                rl.append(rolls[k]); vl.append(vs[k]); aal.append(as_[k])
            aw = np.asarray(al[-CONTEXT:], np.float32)
            rw = np.asarray(rl[-CONTEXT:], np.float32)
            vw = np.asarray(vl[-CONTEXT:], np.float32)
            aaw = np.asarray(aal[-CONTEXT:], np.float32)
            lw = np.asarray(ll[-CONTEXT:], np.float32)
            pe = self._expected1(aw, rw, vw, aaw, lw)
            prev = ll[-1]
            pe = float(np.clip(pe, prev - MAX_DELTA, prev + MAX_DELTA))
            ll.append(pe)
            out[k] = pe
        return out

    def _cost(self, lat, plan, refs, wj, w_du):
        prev = self.lat_h[-1]
        dlat = lat - np.concatenate([[prev], lat[:-1]])
        a_prev = self.act_h[-1]
        da = np.asarray(plan) - np.concatenate([[a_prev], np.asarray(plan)[:-1]])
        return float(np.sum((lat - refs) ** 2) + wj * np.sum(dlat ** 2) + w_du * np.sum(da ** 2))

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        p = self.p
        self.step += 1
        idx = FIRST_UPDATE + self.step - 1

        self.roll_h.append(state.roll_lataccel)
        self.v_h.append(state.v_ego)
        self.a_h.append(state.a_ego)
        self.lat_h.append(current_lataccel)

        if idx < CONTROL_START or len(self.act_h) < CONTEXT - 1:
            u = self._seed_action(target_lataccel, state.roll_lataccel, state.v_ego)
            self.act_h.append(u)
            return u

        H = int(p["H"]); wj = p["w_jerk"]; w_du = p["w_du"]

        def pad(seq, n, fb):
            seq = list(seq[:n])
            return seq + [seq[-1] if seq else fb] * max(0, n - len(seq))
        # per-horizon-step roll/v/a: step k=0 is current (state), k>=1 from future_plan[k-1]
        rolls = [state.roll_lataccel] + pad(future_plan.roll_lataccel, H - 1, state.roll_lataccel)
        vs = [state.v_ego] + pad(future_plan.v_ego, H - 1, state.v_ego)
        as_ = [state.a_ego] + pad(future_plan.a_ego, H - 1, state.a_ego)
        raw = [target_lataccel] + list(future_plan.lataccel)
        refs = np.array([raw[k] if k < len(raw) else raw[-1] for k in range(H)])

        if self.plan is None or len(self.plan) != H:
            self.plan = np.full(H, self._seed_action(target_lataccel, state.roll_lataccel, state.v_ego))
        plan = self.plan.copy()

        def cost_of(pl):
            return self._cost(self._rollout(pl, rolls, vs, as_, H), pl, refs, wj, w_du)

        # coordinate descent with golden-section per action
        for _ in range(int(p["sweeps"])):
            for j in range(H):
                lo, hi = STEER_RANGE
                c1 = hi - _GR * (hi - lo); c2 = lo + _GR * (hi - lo)
                pl = plan.copy()
                pl[j] = c1; f1 = cost_of(pl)
                pl[j] = c2; f2 = cost_of(pl)
                for _ in range(int(p["n_gs"])):
                    if f1 < f2:
                        hi, c2, f2 = c2, c1, f1
                        c1 = hi - _GR * (hi - lo); pl[j] = c1; f1 = cost_of(pl)
                    else:
                        lo, c1, f1 = c1, c2, f2
                        c2 = lo + _GR * (hi - lo); pl[j] = c2; f2 = cost_of(pl)
                plan[j] = 0.5 * (lo + hi)

        u = float(np.clip(plan[0], *STEER_RANGE))
        self.act_h.append(u)
        self.plan = np.concatenate([plan[1:], plan[-1:]])
        return u
