"""Validate that our expected-value window prediction matches the simulator.

Runs the real sim with ff_pid3, then for control steps reconstructs the EXACT
window the simulator used and compares our expected-value prediction to the
realized lataccel. If our predictor is correct, predictions track realized values.
"""
import sys
import numpy as np
import onnxruntime as ort
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import tinyphysics as tp
import os
os.environ.setdefault("FF_PID_PARAMS", "{}")

CONTEXT = 20
BINS = np.linspace(-5, 5, 1024)


def expected(sess, acts, roll, v, a, lats, temp):
    states = np.stack([acts, roll, v, a], axis=1)[None].astype(np.float32)
    toks = np.digitize(np.clip(lats, -5, 5), BINS, right=True)[None].astype(np.int64)
    logits = sess.run(None, {"states": states, "tokens": toks})[0]
    z = logits[0, -1] / temp
    e = np.exp(z - z.max())
    return float((e / e.sum()) @ BINS)


if __name__ == "__main__":
    seg = sys.argv[1] if len(sys.argv) > 1 else "03000"
    model = tp.TinyPhysicsModel(str(ROOT / "models/tinyphysics.onnx"), debug=False)
    ctrl = __import__("controllers.ff_pid3", fromlist=["Controller"]).Controller()
    sim = tp.TinyPhysicsSimulator(model, str(ROOT / f"data/{seg}.csv"), controller=ctrl, debug=False)
    sim.rollout()

    sh = np.array(sim.state_history)          # (N,3): roll, v, a
    ah = np.array(sim.action_history)         # (N,)
    lh = np.array(sim.current_lataccel_history)  # (N,)
    sess = model.ort_session

    errs, preds, reals = [], [], []
    for t in range(100, 400):
        acts = ah[t - 19:t + 1]
        roll = sh[t - 19:t + 1, 0]; v = sh[t - 19:t + 1, 1]; a = sh[t - 19:t + 1, 2]
        past = lh[t - 20:t]
        pe = expected(sess, acts, roll, v, a, past, 0.8)
        pe = np.clip(pe, lh[t - 1] - 0.5, lh[t - 1] + 0.5)
        preds.append(pe); reals.append(lh[t]); errs.append(pe - lh[t])
    errs = np.array(errs)
    preds = np.array(preds); reals = np.array(reals)
    print(f"seg {seg}: pred vs realized over steps 100-399")
    print(f"  RMS(pred-realized) = {np.sqrt(np.mean(errs**2)):.4f}")
    print(f"  corr = {np.corrcoef(preds, reals)[0,1]:.4f}")
    print(f"  mean|realized step-delta| = {np.mean(np.abs(np.diff(reals))):.4f}")
    print(f"  bias = {errs.mean():.4f}")
