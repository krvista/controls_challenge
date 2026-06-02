from . import BaseController
import json
import os
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Correct multi-step shooting MPC.
# Maintains the simulator's exact 20-step autoregressive windows. Each control step,
# searches a single steer value (held over an H-step horizon) that minimizes the
# horizon tracking+jerk cost, rolling the EXPECTED-VALUE model forward with predicted
# lataccel fed back as tokens (on-manifold). This fixes the two failure modes of the
# earlier attempts: (1) 1-step lag saturation -> horizon absorbs lag; (2) off-manifold
# steady-state probe -> consistent token feedback keeps it on-manifold.

CONTEXT = 20
VOCAB = 1024
LAT_RANGE = (-5.0, 5.0)
STEER_RANGE = (-2.0, 2.0)
MAX_DELTA = 0.5
TEMP = 0.8
CONTROL_START = 100
FIRST_UPDATE = CONTEXT
_BINS = np.linspace(LAT_RANGE[0], LAT_RANGE[1], VOCAB)
_GR = 0.6180339887498949  # golden ratio conjugate

DEFAULTS = {
    "H": 6,            # horizon steps
    # real cost per-sample weights: tracking 5000*err^2, jerk 10000*dlat^2 => ratio 2:1.
    "w_jerk": 2.0,     # jerk weight relative to tracking (matches the true objective)
    "n_iter": 9,       # golden-section iterations
    "kp": 0.0,         # small residual feedback (off by default; MPC optimizes cost directly)
    "ki": 0.0,
    "ref_sigma": 0.5,  # mild reference smoothing
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
        m = np.load(_DIR / "ff_model.npz")  # affine inverse for warmup seeding
        self.iv, self.ia, self.ic = m["v_centers"], m["a_v"], m["c_v"]
        self.roll_h, self.v_h, self.a_h, self.act_h, self.lat_h = [], [], [], [], []
        self.error_integral = 0.0
        self.step = 0
        s = max(self.p["ref_sigma"], 1e-3)
        offs = np.arange(0, int(np.ceil(3 * s)) + 1)
        self._w = np.exp(-0.5 * (offs / s) ** 2)
        self._w /= self._w.sum()

    def _seed_action(self, target, roll, v):
        a = np.interp(v, self.iv, self.ia)
        c = np.interp(v, self.iv, self.ic)
        return float(np.clip(a * (target - roll) + c, *STEER_RANGE))

    def _expected(self, acts, roll, v, a, lats):
        states = np.stack([np.asarray(acts[-CONTEXT:], np.float32),
                           np.asarray(roll[-CONTEXT:], np.float32),
                           np.asarray(v[-CONTEXT:], np.float32),
                           np.asarray(a[-CONTEXT:], np.float32)], axis=1)[None]
        toks = np.digitize(np.clip(lats[-CONTEXT:], *LAT_RANGE), _BINS, right=True)[None].astype(np.int64)
        logits = self.sess.run(None, {"states": states.astype(np.float32), "tokens": toks})[0]
        z = logits[0, -1] / TEMP
        e = np.exp(z - z.max())
        return float((e / e.sum()) @ _BINS)

    def _rollout(self, cand, froll, fv, fa, H):
        """Predict lataccel for steps t..t+H-1 holding action=cand (consistent rollout)."""
        al = list(self.act_h); rl = list(self.roll_h); vl = list(self.v_h)
        aal = list(self.a_h); ll = list(self.lat_h)
        preds = []
        for k in range(H):
            if k > 0:
                rl.append(froll[k - 1]); vl.append(fv[k - 1]); aal.append(fa[k - 1])
            al.append(cand)
            prev = ll[-1]
            pk = self._expected(al, rl, vl, aal, ll)
            pk = float(np.clip(pk, prev - MAX_DELTA, prev + MAX_DELTA))
            ll.append(pk); preds.append(pk)
        return preds

    def _cost(self, cand, refs, froll, fv, fa, H):
        preds = self._rollout(cand, froll, fv, fa, H)
        prev = self.lat_h[-1]
        c = 0.0
        wj = self.p["w_jerk"]
        for k in range(H):
            c += (preds[k] - refs[k]) ** 2 + wj * (preds[k] - prev) ** 2
            prev = preds[k]
        return c

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        p = self.p
        self.step += 1
        idx = FIRST_UPDATE + self.step - 1
        if idx == CONTROL_START:
            self.error_integral = 0.0

        self.roll_h.append(state.roll_lataccel)
        self.v_h.append(state.v_ego)
        self.a_h.append(state.a_ego)
        self.lat_h.append(current_lataccel)

        if idx < CONTROL_START or len(self.act_h) < CONTEXT - 1:
            u = self._seed_action(target_lataccel, state.roll_lataccel, state.v_ego)
            self.act_h.append(u)
            return u

        H = int(p["H"])
        fut = future_plan.lataccel
        froll = future_plan.roll_lataccel
        fv = future_plan.v_ego
        fa = future_plan.a_ego
        # reference per horizon step (k=0 is current target), lightly smoothed for jerk
        raw = [target_lataccel] + list(fut[:H])
        refs = []
        for k in range(H):
            seg = raw[k:k + len(self._w)]
            w = self._w[:len(seg)]
            refs.append(float(np.dot(seg, w) / w.sum()))
        # pad future state arrays if near end of segment
        need = H - 1
        froll = list(froll[:need]) + [froll[-1] if froll else state.roll_lataccel] * max(0, need - len(froll))
        fv = list(fv[:need]) + [fv[-1] if fv else state.v_ego] * max(0, need - len(fv))
        fa = list(fa[:need]) + [fa[-1] if fa else state.a_ego] * max(0, need - len(fa))

        # golden-section minimize cost over cand in [-2, 2]
        lo, hi = STEER_RANGE
        a, b = lo, hi
        c1 = b - _GR * (b - a)
        c2 = a + _GR * (b - a)
        f1 = self._cost(c1, refs, froll, fv, fa, H)
        f2 = self._cost(c2, refs, froll, fv, fa, H)
        for _ in range(int(p["n_iter"])):
            if f1 < f2:
                b, c2, f2 = c2, c1, f1
                c1 = b - _GR * (b - a)
                f1 = self._cost(c1, refs, froll, fv, fa, H)
            else:
                a, c1, f1 = c1, c2, f2
                c2 = a + _GR * (b - a)
                f2 = self._cost(c2, refs, froll, fv, fa, H)
        u_mpc = 0.5 * (a + b)

        error = target_lataccel - current_lataccel
        self.error_integral += error
        u_fb = p["kp"] * error + p["ki"] * self.error_integral

        u = float(np.clip(u_mpc + u_fb, *STEER_RANGE))
        self.act_h.append(u)
        return u
