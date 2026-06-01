from . import BaseController
import json
import os
import numpy as np
from pathlib import Path

# Like ff_pid3 but uses the SIM-CALIBRATED one-step inverse (ff_model_v3.npz):
#   steer = g(target, current_lataccel, roll, v_ego, a_ego)
# Features per v-bin: [target, current, roll, target*|target|, a, 1].
DEFAULTS = {
    "ff_gain": 1.0,
    "ref_center": 2.86,
    "ref_sigma": 1.99,
    "kp": 0.11,
    "ki": 0.14,
    "kd": -0.004,
    "i_clip": 1.1,
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
        m = np.load(_DIR / "ff_model_v3.npz")
        self.v_centers = m["v_centers"]
        self.coefs = m["coefs"]  # (n_bins, 6): [target, current, roll, tgt|tgt|, a, 1]
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.step = 0
        c, s = self.p["ref_center"], max(self.p["ref_sigma"], 1e-3)
        W = int(np.ceil(c + 3 * s))
        offs = np.arange(0, W + 1)
        self._w = np.exp(-0.5 * ((offs - c) / s) ** 2)
        self._w /= self._w.sum()

    def _ff(self, target, current, roll, v_ego, a_ego):
        c = np.array([np.interp(v_ego, self.v_centers, self.coefs[:, j]) for j in range(self.coefs.shape[1])])
        feat = np.array([target, current, roll, target * abs(target), a_ego, 1.0])
        return float(feat @ c)

    def _smoothed_ref(self, target_lataccel, future_lat):
        seq = np.concatenate([[target_lataccel], np.asarray(future_lat, dtype=float)])
        n = min(len(self._w), len(seq))
        w = self._w[:n]
        return float((seq[:n] * w).sum() / w.sum())

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        p = self.p
        self.step += 1
        if self.step == 100 - 20 + 1:
            self.error_integral = 0.0
            self.prev_error = 0.0

        ref = self._smoothed_ref(target_lataccel, future_plan.lataccel)
        steer_ff = p["ff_gain"] * self._ff(ref, current_lataccel, state.roll_lataccel, state.v_ego, state.a_ego)

        error = target_lataccel - current_lataccel
        self.error_integral += error
        self.error_integral = float(np.clip(self.error_integral, -p["i_clip"] / max(p["ki"], 1e-6),
                                             p["i_clip"] / max(p["ki"], 1e-6)))
        error_diff = error - self.prev_error
        self.prev_error = error
        steer_fb = p["kp"] * error + p["ki"] * self.error_integral + p["kd"] * error_diff

        return float(np.clip(steer_ff + steer_fb, -2, 2))
