from . import BaseController
import json
import os
import numpy as np
from pathlib import Path

# Same tunable interface as ff_pid, but uses the richer inverse model
# (separate roll coef + saturation term) from ff_model_v2.npz.
DEFAULTS = {
    "ff_gain": 1.0,
    "lookahead": 3,
    "kp": 0.16,
    "ki": 0.12,
    "kd": -0.006,
    "i_clip": 1.5,
    "ref_ema": 0.4,
}

_DIR = Path(__file__).resolve().parent


def _load_params():
    p = dict(DEFAULTS)
    env = os.environ.get("FF_PID_PARAMS")
    if env:
        p.update(json.loads(env))
    return p


class Controller(BaseController):
    """FF (rich speed-scheduled inverse model) + lookahead + PID feedback."""

    def __init__(self):
        self.p = _load_params()
        m = np.load(_DIR / "ff_model_v2.npz")
        self.v_centers = m["v_centers"]
        self.coefs = m["coefs"]  # (n_bins, 5): [lat, roll, lat|lat|, a_ego, 1]
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.ref_filt = None
        self.step = 0

    def _coef(self, v_ego):
        return np.array([np.interp(v_ego, self.v_centers, self.coefs[:, j]) for j in range(self.coefs.shape[1])])

    def _ff(self, lat_target, roll, v_ego, a_ego):
        c = self._coef(v_ego)
        feat = np.array([lat_target, roll, lat_target * abs(lat_target), a_ego, 1.0])
        return float(feat @ c)

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        p = self.p
        self.step += 1
        if self.step == 100 - 20 + 1:
            self.error_integral = 0.0
            self.prev_error = 0.0

        fut = future_plan.lataccel
        k = int(p["lookahead"])
        ref = fut[min(k, len(fut)) - 1] if (fut and k > 0) else target_lataccel

        if p["ref_ema"] > 0:
            if self.ref_filt is None:
                self.ref_filt = ref
            self.ref_filt = (1 - p["ref_ema"]) * ref + p["ref_ema"] * self.ref_filt
            ref = self.ref_filt

        steer_ff = p["ff_gain"] * self._ff(ref, state.roll_lataccel, state.v_ego, state.a_ego)

        error = target_lataccel - current_lataccel
        self.error_integral += error
        self.error_integral = float(np.clip(self.error_integral, -p["i_clip"] / max(p["ki"], 1e-6),
                                             p["i_clip"] / max(p["ki"], 1e-6)))
        error_diff = error - self.prev_error
        self.prev_error = error
        steer_fb = p["kp"] * error + p["ki"] * self.error_integral + p["kd"] * error_diff

        return float(np.clip(steer_ff + steer_fb, -2, 2))
