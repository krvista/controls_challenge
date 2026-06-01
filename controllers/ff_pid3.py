from . import BaseController
import json
import os
import numpy as np
from pathlib import Path

# Consolidated design:
#  - rich speed-scheduled inverse model (ff_model_v2.npz)
#  - FORWARD-LOOKING reference smoothing over the known future plan: a Gaussian
#    window centered `ref_center` steps ahead with width `ref_sigma`. Because it
#    uses future targets, it lowers jerk WITHOUT the lag a causal EMA introduces.
#  - PID feedback on the true tracking error, integral reset at control start.
# Defaults = best params from CMA-ES tuning on a representative spread set
# (val-1000 total_cost = 54.30; lataccel 0.725, jerk 18.07).
DEFAULTS = {
    "ff_gain": 0.7916510351789442,
    "ref_center": 3.981853870138741,   # window center, steps ahead (phase lead)
    "ref_sigma": 1.5758175214224928,   # window width (smoothing strength)
    "kp": 0.07651180124198706,
    "ki": 0.10516086467551312,
    "kd": 0.049513137282490155,
    "i_clip": 1.3973288915902329,
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
        m = np.load(_DIR / "ff_model_v2.npz")
        self.v_centers = m["v_centers"]
        self.coefs = m["coefs"]  # (n_bins, 5): [lat, roll, lat|lat|, a_ego, 1]
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.step = 0
        # precompute Gaussian weights over offsets [0..W]
        c, s = self.p["ref_center"], max(self.p["ref_sigma"], 1e-3)
        W = int(np.ceil(c + 3 * s))
        offs = np.arange(0, W + 1)
        self._w = np.exp(-0.5 * ((offs - c) / s) ** 2)
        self._w /= self._w.sum()

    def _ff(self, lat_target, roll, v_ego, a_ego):
        c = np.array([np.interp(v_ego, self.v_centers, self.coefs[:, j]) for j in range(self.coefs.shape[1])])
        feat = np.array([lat_target, roll, lat_target * abs(lat_target), a_ego, 1.0])
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
        steer_ff = p["ff_gain"] * self._ff(ref, state.roll_lataccel, state.v_ego, state.a_ego)

        error = target_lataccel - current_lataccel
        self.error_integral += error
        self.error_integral = float(np.clip(self.error_integral, -p["i_clip"] / max(p["ki"], 1e-6),
                                             p["i_clip"] / max(p["ki"], 1e-6)))
        error_diff = error - self.prev_error
        self.prev_error = error
        steer_fb = p["kp"] * error + p["ki"] * self.error_integral + p["kd"] * error_diff

        return float(np.clip(steer_ff + steer_fb, -2, 2))
