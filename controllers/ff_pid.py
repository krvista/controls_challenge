from . import BaseController
import json
import os
import numpy as np
from pathlib import Path

# Hyperparameters are tunable via the FF_PID_PARAMS env var (JSON) so the CMA-ES
# tuner can evaluate candidates with the parallel rollout harness.
DEFAULTS = {
    "ff_gain": 1.0,      # global scale on the feedforward steer
    "lookahead": 2,      # use future_plan.lataccel[lookahead] as the FF target (phase lead)
    "kp": 0.05,          # feedback proportional gain on residual error
    "ki": 0.02,          # feedback integral gain
    "kd": -0.01,         # feedback derivative gain
    "i_clip": 1.0,       # integral clamp
    "ref_ema": 0.0,      # EMA smoothing of the reference (0 = none)
}

_DIR = Path(__file__).resolve().parent
ACC_G = 9.81


def _load_params():
    p = dict(DEFAULTS)
    env = os.environ.get("FF_PID_PARAMS")
    if env:
        p.update(json.loads(env))
    return p


class Controller(BaseController):
    """Feedforward (speed-scheduled inverse model) + lookahead + PID feedback."""

    def __init__(self):
        self.p = _load_params()
        m = np.load(_DIR / "ff_model.npz")
        self.v_centers = m["v_centers"]
        self.a_v = m["a_v"]
        self.b_v = m["b_v"]
        self.c_v = m["c_v"]
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.ref_filt = None
        self.step = 0

    def _ff(self, lat_target, roll, v_ego, a_ego):
        a = np.interp(v_ego, self.v_centers, self.a_v)
        b = np.interp(v_ego, self.v_centers, self.b_v)
        c = np.interp(v_ego, self.v_centers, self.c_v)
        return a * (lat_target - roll) + b * a_ego + c

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        p = self.p
        self.step += 1
        # update() is first called at sim idx CONTEXT_LENGTH(20); control starts at
        # idx CONTROL_START_IDX(100) => the 81st call. Reset feedback state there so
        # warmup steps don't wind up the integrator.
        if self.step == 100 - 20 + 1:
            self.error_integral = 0.0
            self.prev_error = 0.0

        # --- reference: lookahead into the known future plan for phase lead ---
        fut = future_plan.lataccel
        k = int(p["lookahead"])
        if fut and k > 0:
            ref = fut[min(k, len(fut)) - 1]
        else:
            ref = target_lataccel

        # optional EMA smoothing of the reference (reduces jerk)
        if p["ref_ema"] > 0:
            if self.ref_filt is None:
                self.ref_filt = ref
            self.ref_filt = (1 - p["ref_ema"]) * ref + p["ref_ema"] * self.ref_filt
            ref = self.ref_filt

        # --- feedforward from inverse model ---
        steer_ff = p["ff_gain"] * self._ff(ref, state.roll_lataccel, state.v_ego, state.a_ego)

        # --- feedback on residual (track the actual current target) ---
        error = target_lataccel - current_lataccel
        self.error_integral += error
        self.error_integral = float(np.clip(self.error_integral, -p["i_clip"] / max(p["ki"], 1e-6),
                                             p["i_clip"] / max(p["ki"], 1e-6)))
        error_diff = error - self.prev_error
        self.prev_error = error
        steer_fb = p["kp"] * error + p["ki"] * self.error_integral + p["kd"] * error_diff

        return float(np.clip(steer_ff + steer_fb, -2, 2))
