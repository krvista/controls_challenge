from . import BaseController
import json
import os
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Receding-horizon MPC via Gauss-Newton on a finite-difference-linearized plant
# ("direct quadratic optimization"). Each control step:
#   1. nominal rollout of a warm-started action plan (consistent, on-manifold).
#   2. finite-diff Jacobian G[k,j] = d lat[k] / d a[j] (lower-triangular, causal).
#   3. residuals match the TRUE cost: tracking (w=1) and jerk (w=2) per sample;
#      solve (Jr^T Jr + lambda I) da = -Jr^T r, line-search, clip. iterate.
#   4. apply plan[0]; shift plan for warm start.
# Predictor verified against the sim (corr 0.9998, RMS 0.008) by tools/check_model.py.

CONTEXT = 20
VOCAB = 1024
LAT_RANGE = (-5.0, 5.0)
STEER_RANGE = (-2.0, 2.0)
MAX_DELTA = 0.5
TEMP = 0.8
CONTROL_START = 100
FIRST_UPDATE = CONTEXT
_BINS = np.linspace(LAT_RANGE[0], LAT_RANGE[1], VOCAB)

DEFAULTS = {
    "H": 20,           # horizon
    "w_jerk": 2.0,     # jerk:tracking weight ratio (matches true cost)
    "n_gn": 2,         # Gauss-Newton iterations per step
    "lam": 0.5,        # Levenberg damping
    "eps": 0.02,       # finite-diff step
    "ref_sigma": 0.0,  # optional reference smoothing (0 = track raw targets)
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
        self.plan = None  # warm-started action sequence

    def _seed_action(self, target, roll, v):
        a = np.interp(v, self.iv, self.ia)
        c = np.interp(v, self.iv, self.ic)
        return float(np.clip(a * (target - roll) + c, *STEER_RANGE))

    def _expected_batch(self, states, tokens):
        logits = self.sess.run(None, {"states": states, "tokens": tokens})[0]
        z = logits[:, -1, :] / TEMP
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        probs = e / e.sum(axis=1, keepdims=True)
        return probs @ _BINS

    def _rollout(self, plan, froll, fv, fa):
        """lat[0..H-1] for the given action plan, consistent on-manifold rollout."""
        H = len(plan)
        al = list(self.act_h); rl = list(self.roll_h); vl = list(self.v_h)
        aal = list(self.a_h); ll = list(self.lat_h)
        out = np.empty(H)
        for k in range(H):
            if k > 0:
                rl.append(froll[k - 1]); vl.append(fv[k - 1]); aal.append(fa[k - 1])
            al.append(plan[k])
            states = np.stack([np.asarray(al[-CONTEXT:], np.float32),
                               np.asarray(rl[-CONTEXT:], np.float32),
                               np.asarray(vl[-CONTEXT:], np.float32),
                               np.asarray(aal[-CONTEXT:], np.float32)], axis=1)[None].astype(np.float32)
            toks = np.digitize(np.clip(np.asarray(ll[-CONTEXT:]), *LAT_RANGE), _BINS, right=True)[None].astype(np.int64)
            pk = float(self._expected_batch(states, toks)[0])
            pk = float(np.clip(pk, ll[-1] - MAX_DELTA, ll[-1] + MAX_DELTA))
            ll.append(pk); out[k] = pk
        return out

    def _cost(self, lat, refs, wj):
        prev = self.lat_h[-1]
        c = 0.0
        for k in range(len(lat)):
            c += (lat[k] - refs[k]) ** 2 + wj * (lat[k] - prev) ** 2
            prev = lat[k]
        return c

    def _jacobian(self, plan, lat0, froll, fv, fa, eps):
        H = len(plan)
        G = np.zeros((H, H))
        for j in range(H):
            pp = plan.copy(); pp[j] += eps
            latp = self._rollout(pp, froll, fv, fa)
            G[j:, j] = (latp[j:] - lat0[j:]) / eps  # causal: a[j] affects lat[k>=j]
        return G

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

        H = int(p["H"])
        wj = p["w_jerk"]
        eps = p["eps"]
        lam = p["lam"]

        # references and future states over the horizon
        fut = list(future_plan.lataccel)
        raw = [target_lataccel] + fut
        refs = np.array([raw[k] if k < len(raw) else raw[-1] for k in range(H)])
        if p["ref_sigma"] > 0:
            s = p["ref_sigma"]; w = np.exp(-0.5 * (np.arange(0, int(3 * s) + 1) / s) ** 2); w /= w.sum()
            refs = np.array([np.dot(raw[k:k + len(w)], w[:len(raw) - k]) / w[:len(raw) - k].sum()
                             if k < len(raw) else raw[-1] for k in range(H)])

        def pad(seq, n, fb):
            seq = list(seq[:n])
            return seq + [seq[-1] if seq else fb] * max(0, n - len(seq))
        froll = pad(future_plan.roll_lataccel, H - 1, state.roll_lataccel)
        fv = pad(future_plan.v_ego, H - 1, state.v_ego)
        fa = pad(future_plan.a_ego, H - 1, state.a_ego)

        # warm start
        if self.plan is None or len(self.plan) != H:
            self.plan = np.full(H, self._seed_action(target_lataccel, state.roll_lataccel, state.v_ego))
        plan = self.plan.copy()

        lat = self._rollout(plan, froll, fv, fa)
        cost = self._cost(lat, refs, wj)
        for _ in range(int(p["n_gn"])):
            G = self._jacobian(plan, lat, froll, fv, fa, eps)
            # residuals: tracking r_t = lat-ref ; jerk r_j = sqrt(wj)*(lat[k]-lat[k-1])
            dlat = lat - np.concatenate([[self.lat_h[-1]], lat[:-1]])
            r = np.concatenate([lat - refs, np.sqrt(wj) * dlat])
            # jacobian of jerk residual: sqrt(wj)*(G[k]-G[k-1]); G[-1]=0
            Gj = np.sqrt(wj) * (G - np.vstack([np.zeros(H), G[:-1]]))
            Jr = np.vstack([G, Gj])  # (2H, H)
            A = Jr.T @ Jr + lam * np.eye(H)
            g = Jr.T @ r
            try:
                da = -np.linalg.solve(A, g)
            except np.linalg.LinAlgError:
                break
            # line search on true cost
            best = (cost, plan, lat)
            for alpha in (1.0, 0.5, 0.25):
                cand = np.clip(plan + alpha * da, *STEER_RANGE)
                latc = self._rollout(cand, froll, fv, fa)
                cc = self._cost(latc, refs, wj)
                if cc < best[0]:
                    best = (cc, cand, latc)
                    break
            if best[0] >= cost - 1e-9:
                break
            cost, plan, lat = best

        u = float(np.clip(plan[0], *STEER_RANGE))
        self.act_h.append(u)
        # warm start next step: shift
        self.plan = np.concatenate([plan[1:], plan[-1:]])
        return u
