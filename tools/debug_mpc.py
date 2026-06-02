"""Isolate the MPC optimizer: seed it with REAL history at a mid-segment step and
inspect the optimized plan, predicted lataccel trajectory, and targets."""
import sys, os, json
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import tinyphysics as tp

os.environ["FF_PID_PARAMS"] = json.dumps({"H": 10, "sweeps": 2, "n_gs": 7})
seg = sys.argv[1] if len(sys.argv) > 1 else "03000"
ctrl_name = sys.argv[2] if len(sys.argv) > 2 else "mpc5"
t = int(sys.argv[3]) if len(sys.argv) > 3 else 200

model = tp.TinyPhysicsModel(str(ROOT / "models/tinyphysics.onnx"), debug=False)
# get real histories via an ff_pid3 rollout
os.environ_bak = os.environ.get("FF_PID_PARAMS")
import importlib
ff = importlib.import_module("controllers.ff_pid3").Controller()
sim = tp.TinyPhysicsSimulator(model, str(ROOT / f"data/{seg}.csv"), controller=ff, debug=False)
sim.rollout()
sh = np.array(sim.state_history); ah = np.array(sim.action_history); lh = np.array(sim.current_lataccel_history)
data = sim.data

# seed a fresh mpc controller with REAL history up to step t
os.environ["FF_PID_PARAMS"] = json.dumps({"H": 10, "sweeps": 2, "n_gs": 7})
C = importlib.import_module(f"controllers.{ctrl_name}").Controller()
C.act_h = list(ah[:t])          # actions 0..t-1
C.roll_h = list(sh[:t + 1, 0])  # roll 0..t  (includes current step t)
C.v_h = list(sh[:t + 1, 1])
C.a_h = list(sh[:t + 1, 2])
C.lat_h = list(lh[:t])          # lat 0..t-1
C.step = t - tp.CONTEXT_LENGTH + 1     # so idx == t

State = tp.State; FuturePlan = tp.FuturePlan
fp = FuturePlan(
    lataccel=data['target_lataccel'].values[t + 1:t + tp.FUTURE_PLAN_STEPS].tolist(),
    roll_lataccel=data['roll_lataccel'].values[t + 1:t + tp.FUTURE_PLAN_STEPS].tolist(),
    v_ego=data['v_ego'].values[t + 1:t + tp.FUTURE_PLAN_STEPS].tolist(),
    a_ego=data['a_ego'].values[t + 1:t + tp.FUTURE_PLAN_STEPS].tolist())
st = State(roll_lataccel=sh[t, 0], v_ego=sh[t, 1], a_ego=sh[t, 2])
target = data['target_lataccel'].values[t]

u = C.update(target, lh[t - 1], st, fp)
H = C.p["H"]
rolls = [st.roll_lataccel] + fp.roll_lataccel[:H - 1]
vs = [st.v_ego] + fp.v_ego[:H - 1]
as_ = [st.a_ego] + fp.a_ego[:H - 1]
# recompute the optimized plan's predicted trajectory (use the warm-started plan AFTER update -> shift back)
# Instead, re-evaluate using C.plan from before apply: reconstruct by re-optimizing print inside is hard; show applied u and a fresh rollout of held-u
print(f"seg {seg} step {t}: target={target:.3f} current={lh[t-1]:.3f} applied_u={u:.3f}")
print(f"  refs(H)   = {np.array([target]+fp.lataccel[:H-1]).round(3)}")
# predicted trajectory if we hold the applied plan (C.plan is the shifted plan; reconstruct pre-shift)
plan_after = C.plan  # shifted: plan[1:]+plan[-1]
pred = C._rollout(np.concatenate([[u], plan_after[:-1]]), rolls, vs, as_, H)
print(f"  pred lat  = {pred.round(3)}")
print(f"  plan(a)   = {np.concatenate([[u], plan_after[:-1]]).round(3)}")
print(f"  realized lat[t..t+5] (ff_pid3 path) = {lh[t:t+6].round(3)}")
