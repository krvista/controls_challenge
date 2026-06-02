from . import BaseController
import json
import os
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Batched Gauss-Newton receding-horizon MPC WITH action-rate regularization.
# Combines mpc4's speed (batched finite-diff Jacobian via the ONNX batch dim) with
# the key stability fix from mpc5's diagnosis: penalizing action rate keeps the plan
# smooth / on-manifold so the optimizer can't exploit the model with adversarial
# oscillation. The regularizer's Jacobian is a constant difference matrix (free).
# Cost = tracking + 2*jerk (true objective) + w_du * action-rate (regularizer).
# This is the recommended controller for the offline full-eval. See HANDOFF notes.

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
    "H": 20,
    "w_jerk": 2.0,
    "w_du": 0.5,       # action-rate regularizer (stability / on-manifold)
    "n_gn": 3,
    "lam": 1.0,
    "eps": 0.02,
    "tr": 0.5,
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
        self.plan = None

    def _seed_action(self, target, roll, v):
        a = np.interp(v, self.iv, self.ia)
        c = np.interp(v, self.iv, self.ic)
        return float(np.clip(a * (target - roll) + c, *STEER_RANGE))

    def _expected(self, states, tokens):
        logits = self.sess.run(None, {"states": states, "tokens": tokens})[0]
        z = logits[:, -1, :] / TEMP
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return (e / e.sum(axis=1, keepdims=True)) @ _BINS

    def _rollout_many(self, Ah, froll, fv, fa):
        B, H = Ah.shape
        RL = list(self.roll_h); VL = list(self.v_h); AL = list(self.a_h)
        A = np.tile(np.asarray(self.act_h, np.float32), (B, 1))
        L = np.tile(np.asarray(self.lat_h, np.float32), (B, 1))
        out = np.empty((B, H))
        for k in range(H):
            if k > 0:
                RL.append(froll[k - 1]); VL.append(fv[k - 1]); AL.append(fa[k - 1])
            A = np.concatenate([A, Ah[:, k:k + 1]], axis=1)
            aw = A[:, -CONTEXT:]
            rw = np.broadcast_to(np.asarray(RL[-CONTEXT:], np.float32), (B, CONTEXT))
            vw = np.broadcast_to(np.asarray(VL[-CONTEXT:], np.float32), (B, CONTEXT))
            aaw = np.broadcast_to(np.asarray(AL[-CONTEXT:], np.float32), (B, CONTEXT))
            states = np.stack([aw, rw, vw, aaw], axis=2).astype(np.float32)
            toks = np.digitize(np.clip(L[:, -CONTEXT:], *LAT_RANGE), _BINS, right=True).astype(np.int64)
            pe = self._expected(states, toks)
            prev = L[:, -1]
            pe = np.clip(pe, prev - MAX_DELTA, prev + MAX_DELTA)
            L = np.concatenate([L, pe[:, None]], axis=1)
            out[:, k] = pe
        return out

    def _cost(self, lat, plan, refs, wj, w_du):
        prev_l = self.lat_h[-1]; prev_a = self.act_h[-1]
        dlat = lat - np.concatenate([[prev_l], lat[:-1]])
        da = np.asarray(plan) - np.concatenate([[prev_a], np.asarray(plan)[:-1]])
        return float(np.sum((lat - refs) ** 2) + wj * np.sum(dlat ** 2) + w_du * np.sum(da ** 2))

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

        H = int(p["H"]); wj = p["w_jerk"]; w_du = p["w_du"]
        eps = p["eps"]; lam = p["lam"]; tr = p["tr"]
        raw = [target_lataccel] + list(future_plan.lataccel)
        refs = np.array([raw[k] if k < len(raw) else raw[-1] for k in range(H)])

        def pad(seq, n, fb):
            seq = list(seq[:n])
            return seq + [seq[-1] if seq else fb] * max(0, n - len(seq))
        froll = pad(future_plan.roll_lataccel, H - 1, state.roll_lataccel)
        fv = pad(future_plan.v_ego, H - 1, state.v_ego)
        fa = pad(future_plan.a_ego, H - 1, state.a_ego)

        if self.plan is None or len(self.plan) != H:
            self.plan = np.full(H, self._seed_action(target_lataccel, state.roll_lataccel, state.v_ego))
        plan = self.plan.copy()

        # constant difference matrix for the action-rate residual du[k]=a[k]-a[k-1]
        D = np.eye(H) - np.eye(H, k=-1)
        a_prev = self.act_h[-1]
        sdu = np.sqrt(w_du); sj = np.sqrt(wj)

        lat = self._rollout_many(plan[None], froll, fv, fa)[0]
        cost = self._cost(lat, plan, refs, wj, w_du)
        I = np.eye(H)
        for _ in range(int(p["n_gn"])):
            Ah = np.tile(plan, (H + 1, 1))
            for j in range(H):
                Ah[1 + j, j] += eps
            latall = self._rollout_many(Ah, froll, fv, fa)
            lat0 = latall[0]
            G = ((latall[1:] - lat0) / eps).T
            dlat = lat0 - np.concatenate([[self.lat_h[-1]], lat0[:-1]])
            du = plan - np.concatenate([[a_prev], plan[:-1]])
            r = np.concatenate([lat0 - refs, sj * dlat, sdu * du])
            Gj = sj * (G - np.vstack([np.zeros(H), G[:-1]]))
            Jr = np.vstack([G, Gj, sdu * D])
            try:
                da = -np.linalg.solve(Jr.T @ Jr + lam * I, Jr.T @ r)
            except np.linalg.LinAlgError:
                break
            da = np.clip(da, -tr, tr)
            alphas = np.array([1.0, 0.6, 0.3, 0.15, 0.07])
            cands = np.clip(plan[None] + alphas[:, None] * da[None], *STEER_RANGE)
            latc = self._rollout_many(cands, froll, fv, fa)
            costs = np.array([self._cost(latc[i], cands[i], refs, wj, w_du) for i in range(len(alphas))])
            bi = int(np.argmin(costs))
            if costs[bi] < cost - 1e-9:
                plan = cands[bi]; lat = latc[bi]; cost = costs[bi]
            else:
                break

        u = float(np.clip(plan[0], *STEER_RANGE))
        self.act_h.append(u)
        self.plan = np.concatenate([plan[1:], plan[-1:]])
        return u
