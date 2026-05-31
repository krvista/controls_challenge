from . import BaseController
import numpy as np


class Controller(BaseController):
  """
  Feedforward + lookahead + PID controller.

  Rationale (cost = 50 * lataccel_cost + jerk_cost):
    - The lataccel (tracking) term is weighted 50x, so reducing tracking error
      is by far the biggest lever. A plain PID is purely reactive and always
      lags the target, which dominates the cost whenever the trajectory has
      real curvature transients.
    - We therefore lead with a *feedforward* term that maps the desired lateral
      acceleration directly to a steer command, using the (roughly linear)
      steer -> lataccel relationship of the simulator model. The PID only has
      to clean up the residual, so its gains (and the resulting jerk) stay low.
    - The feedforward target is blended slightly into the future via the
      `future_plan`, which compensates for the actuator/vehicle lag so the
      steering leads the target instead of chasing it.

  The effective feedforward gain (FF_GAIN) and the lookahead horizon were
  calibrated against the tinyphysics model. With a good feedforward the loop
  tracks aggressive maneuvers far better than PID while keeping jerk comparable.
  """

  # --- feedforward ---
  FF_GAIN = 2.4      # effective lataccel produced per unit steer (closed-loop calibrated)
  LOOK = 10          # lookahead horizon in frames (~1.0s at 10 FPS)
  LOOK_W = 0.3       # weight of the future target blended into the feedforward target

  # --- pid residual ---
  KP = 0.20
  KI = 0.05
  KD = -0.10
  I_CLIP = 5.0       # integral anti-windup clamp

  def __init__(self):
    self.error_integral = 0.0
    self.prev_error = 0.0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    roll = state.roll_lataccel

    # Lookahead feedforward target: lead the trajectory to cancel actuation lag.
    future = future_plan.lataccel
    if len(future) > self.LOOK:
      ff_target = (1.0 - self.LOOK_W) * target_lataccel + self.LOOK_W * future[self.LOOK]
    else:
      ff_target = target_lataccel

    # Feedforward: invert the steer -> lataccel map. The road-roll component is
    # already reflected in current_lataccel, so we command the remainder.
    feedforward = (ff_target - roll) / self.FF_GAIN

    # PID feedback on the residual tracking error.
    error = target_lataccel - current_lataccel
    self.error_integral = np.clip(self.error_integral + error, -self.I_CLIP, self.I_CLIP)
    error_diff = error - self.prev_error
    self.prev_error = error
    feedback = self.KP * error + self.KI * self.error_integral + self.KD * error_diff

    return feedforward + feedback
