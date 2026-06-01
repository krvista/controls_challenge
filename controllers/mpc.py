from . import BaseController
import json
import os
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Online 1-step inversion of the ACTUAL tinyphysics ONNX plant.
# Each control step we bisect the steer action so the model's EXPECTED next
# lataccel (probability-weighted decode at the sim's temperature) equals a
# forward-smoothed reference. Uses the true plant => no fit-extrapolation error.
#
# History handling: the controller maintains its own 20-step windows of
# (action, roll, v_ego, a_ego) and observed lataccels, matching the simulator's
# get_current_lataccel inputs. Warmup actions (unknown logged commands) are seeded
# from a cheap inverse estimate; they wash out within ~20 control steps.

ACC_G = 9.81
CONTEXT = 20
VOCAB = 1024
LAT_RANGE = (-5.0, 5.0)
STEER_RANGE = (-2.0, 2.0)
MAX_DELTA = 0.5
TEMP = 0.8
CONTROL_START = 100
FIRST_UPDATE = CONTEXT  # sim calls update first at step_idx == CONTEXT (20)

DEFAULTS = {
    "ref_center": 2.5,
    "ref_sigma": 1.6,
    "kp": 0.04,
    "ki": 0.04,
    "kd": 0.0,
    "i_clip": 0.5,
    "n_iter": 6,        # bisection iterations
}

_DIR = Path(__file__).resolve().parent
_BINS = np.linspace(LAT_RANGE[0], LAT_RANGE[1], VOCAB)


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
        # cheap real-data inverse for warmup action seeding / fallback
        m = np.load(_DIR / "ff_model.npz")
        self.iv, self.ia, self.ic = m["v_centers"], m["a_v"], m["c_v"]
        # windows
        self.roll_h, self.v_h, self.a_h = [], [], []
        self.act_h = []
        self.lat_h = []
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.step = 0
        c, s = self.p["ref_center"], max(self.p["ref_sigma"], 1e-3)
        W = int(np.ceil(c + 3 * s))
        offs = np.arange(0, W + 1)
        self._w = np.exp(-0.5 * ((offs - c) / s) ** 2)
        self._w /= self._w.sum()

    def _seed_action(self, target, roll, v):
        a = np.interp(v, self.iv, self.ia)
        c = np.interp(v, self.iv, self.ic)
        return float(np.clip(a * (target - roll) + c, *STEER_RANGE))

    def _smoothed_ref(self, target, future_lat):
        seq = np.concatenate([[target], np.asarray(future_lat, dtype=float)])
        n = min(len(self._w), len(seq))
        w = self._w[:n]
        return float((seq[:n] * w).sum() / w.sum())

    def _predict_expected(self, cand_action):
        """Expected next lataccel via the real model. Steady-state probe: hold
        cand_action across the whole action window so we invert the model's
        steady-state gain (avoids the lag ill-posedness of a 1-step probe)."""
        if self.p.get("ss_mode", 1):
            acts = np.full(CONTEXT, cand_action, dtype=np.float32)
        else:
            acts = np.array(self.act_h[-(CONTEXT - 1):] + [cand_action], dtype=np.float32)
        roll = np.array(self.roll_h[-CONTEXT:], dtype=np.float32)
        v = np.array(self.v_h[-CONTEXT:], dtype=np.float32)
        a = np.array(self.a_h[-CONTEXT:], dtype=np.float32)
        states = np.stack([acts, roll, v, a], axis=1)[None].astype(np.float32)  # (1,20,4)
        lats = np.clip(np.array(self.lat_h[-CONTEXT:]), *LAT_RANGE)
        tokens = np.digitize(lats, _BINS, right=True)[None].astype(np.int64)    # (1,20)
        logits = self.sess.run(None, {"states": states, "tokens": tokens})[0]
        z = logits[0, -1] / TEMP
        e = np.exp(z - z.max())
        probs = e / e.sum()
        return float(probs @ _BINS)

    def _invert(self, ref, current):
        """Bisect steer so the model's (steady-state) expected lataccel ~= ref."""
        if not self.p.get("ss_mode", 1):
            ref = float(np.clip(ref, current - MAX_DELTA, current + MAX_DELTA))
        lo, hi = STEER_RANGE
        flo = self._predict_expected(lo) - ref
        fhi = self._predict_expected(hi) - ref
        if flo > 0:  # even min steer overshoots ref
            return lo
        if fhi < 0:  # even max steer undershoots
            return hi
        for _ in range(int(self.p["n_iter"])):
            mid = 0.5 * (lo + hi)
            fm = self._predict_expected(mid) - ref
            if fm > 0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        p = self.p
        self.step += 1
        idx = FIRST_UPDATE + self.step - 1  # sim step_idx for this call
        if idx == CONTROL_START:
            self.error_integral = 0.0
            self.prev_error = 0.0

        # record windows. current_lataccel is the realized lataccel of the prev action.
        self.roll_h.append(state.roll_lataccel)
        self.v_h.append(state.v_ego)
        self.a_h.append(state.a_ego)
        self.lat_h.append(current_lataccel)

        if idx < CONTROL_START:
            # warmup: sim ignores our output; seed an estimated action for the window
            u = self._seed_action(target_lataccel, state.roll_lataccel, state.v_ego)
            self.act_h.append(u)
            return u

        ref = self._smoothed_ref(target_lataccel, future_plan.lataccel)
        # need a full 20-window; if not yet (shouldn't happen at idx>=100), fall back
        if len(self.act_h) < CONTEXT - 1:
            u = self._seed_action(ref, state.roll_lataccel, state.v_ego)
            self.act_h.append(u)
            return u

        u_ff = self._invert(ref, current_lataccel)

        # light feedback on residual tracking error for robustness
        error = target_lataccel - current_lataccel
        self.error_integral += error
        self.error_integral = float(np.clip(self.error_integral, -p["i_clip"] / max(p["ki"], 1e-6),
                                             p["i_clip"] / max(p["ki"], 1e-6)))
        error_diff = error - self.prev_error
        self.prev_error = error
        u_fb = p["kp"] * error + p["ki"] * self.error_integral + p["kd"] * error_diff

        u = float(np.clip(u_ff + u_fb, *STEER_RANGE))
        self.act_h.append(u)
        return u
