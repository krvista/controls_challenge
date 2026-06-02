# Model-based MPC controllers — handoff notes

This documents the per-segment model-based controllers built for the comma controls
challenge, intended to be run **offline** (they are compute-heavy; the top leaderboard
entries are explicitly "much compute"). The non-model feedforward controller
(`controllers/ff_pid3.py`, total_cost ≈ 59 on the full 5000-seg eval) is the fast,
robust fallback; the MPC controllers below target the top of the leaderboard.

## Approach

The simulator's plant is the ONNX model in `models/tinyphysics.onnx`. We reproduce its
expected-value prediction inside the controller (sampling replaced by the
probability-weighted decode `softmax(logits/0.8) · bins`). This predictor was **verified**
against the real simulator — `python tools/check_model.py 03000` → corr 0.9998,
RMS 0.008. So the model is an accurate, differentiable-by-finite-difference plant.

Each control step we solve a receding-horizon optimal control problem over an `H`-step
window, minimizing the **true cost** (tracking + 2·jerk; weights derived exactly:
total_cost ∝ Σ (lat−tgt)² + 2·(Δlat)²), roll the predictor forward on-manifold (feeding
predicted lataccel back as tokens), apply the first action, and warm-start the next step.
The full future target trajectory needed comes from `future_plan` (≈5 s lookahead).

## THE key stability fix (action regularization)

Naively optimizing the action sequence **diverges** (jerk explodes ~1000×). Root cause,
found via `tools/debug_mpc.py`: the optimizer **exploits the model off-manifold** — it
finds wildly oscillating action sequences that the model (trained on smooth human driving)
maps to a deceptively smooth lataccel, but the real sim turns into violent jerk.

Fix: add an **action-rate penalty** `w_du · Σ(a[k]−a[k−1])²` to the objective. This keeps
the plan smooth / in-distribution, so the model's predictions stay valid. With it, plans
are smooth and the closed loop is stable (verified: seg 03000 stable, no divergence).
`w_du` is the critical knob: too low → off-manifold divergence; too high → under-actuated
(tracks no better than the FF controller). Sweet spot ≈ 0.3–1.0.

## Controllers

- **`controllers/mpc6.py` (RECOMMENDED):** batched Gauss-Newton. Finite-difference plant
  Jacobian computed by batching all H+1 perturbed rollouts through the ONNX batch
  dimension; GN step with Levenberg damping + trust region + line search; action-rate
  regularization added as a constant-Jacobian residual. Fastest of the MPCs.
- **`controllers/mpc5.py`:** coordinate descent with golden-section line search per action
  (no Jacobian). Simplest/most robust reference implementation; slower.
- `controllers/mpc2.py`, `mpc3.py`, `mpc4.py`: earlier iterations kept for the record
  (held-action / un-regularized GN — superseded; mpc3/mpc4 diverge without `w_du`).

Hyperparameters via the `FF_PID_PARAMS` env var (JSON), e.g.
`FF_PID_PARAMS='{"H":20,"n_gn":3,"w_du":0.5}'`.

| param | meaning | speed↔quality |
|---|---|---|
| `H` | horizon steps | ↑H = better, slower |
| `w_du` | action-rate regularizer | tune for stability (≈0.3–1.0) |
| `n_gn` | GN iterations/step (mpc6) | ↑ = better, slower |
| `sweeps`,`n_gs` | coord-descent depth (mpc5) | ↑ = better, slower |

## Running the full evaluation (offline)

```
# single-segment debug
python tinyphysics.py --model_path ./models/tinyphysics.onnx \
  --data_path ./data/03000.csv --controller mpc6

# small validation batch (set params via env)
FF_PID_PARAMS='{"H":20,"n_gn":3,"w_du":0.5}' \
python tools/evalutil.py --controller mpc6 --split custom --start 3000 --n 50

# official leaderboard report (5000 segs) — COMPUTE HEAVY (hours+; run offline)
FF_PID_PARAMS='{"H":20,"n_gn":3,"w_du":0.5}' \
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data \
  --num_segs 5000 --test_controller mpc6 --baseline_controller pid
```

Note: `eval.py`/`tinyphysics.py` read `FF_PID_PARAMS` because the worker processes are
forked after the env var is set; for a fixed final submission, bake the chosen params
into `mpc6.py`'s `DEFAULTS`.

## Speed notes

Cost is dominated by ONNX inferences: ~`(1 + 2·n_gn)` rollouts × `H` steps per control
step × ~400 control steps per segment. On this CPU that is ~minutes/segment. To run 5000
segments, use a machine with many cores (the harness uses 16 workers; increase
`max_workers`), and/or reduce `H`/`n_gn`. Quality/runtime should be tuned on a few hundred
representative segments (see `tools/evalutil.py` splits) before the full run.

## Empirical status & the core tension (read this)

Measured on validation segments (held-out 3000+):
- `ff_pid3` (FF baseline): stable, total ≈ 56–59, jerk ≈ 24–25.
- `mpc5` (coord-descent, `w_du`≈1.0): **stable** (no divergence) but **slow** (~10–12 min/seg);
  on straight segments ≈ ff_pid3, with lower jerk.
- `mpc6` (batched GN, `w_du`=0.5): faster-ish but **GN still diverges on a minority of
  segments** (p95 blows up) — needs higher `w_du`/`lam` to fully stabilize, which then
  caps its tracking advantage.

**Fundamental tension:** the model is accurate only *on-manifold* (smooth, human-like
actions — verified corr 0.9998 there). Large tracking-error reductions require *aggressive*
actions, exactly where the model is least reliable and the optimizer is tempted to exploit
it. The `w_du` regularization that prevents divergence also limits how aggressive the plan
can be, so a naive MPC does **not** robustly beat the well-tuned FF without more work.

Paths to actually win (for offline iteration):
1. **Per-segment trust-region tuning of `w_du`/`lam`** — start high (stable) and relax only
   while the realized (not predicted) cost keeps dropping.
2. **Penalize deviation of predicted-from-realized** (re-predict the just-applied action and
   add the mismatch as a model-trust penalty) to detect off-manifold exploitation online.
3. **Prefer `mpc5`'s coordinate descent** (more robust than GN) and invest compute (longer
   horizon, more sweeps) — this is the safer base to push below ff_pid3.
4. Match the leaderboard leaders' "much compute": many cores + long runtime; consider
   exporting the model to ONNX-GPU / batching across segments.

## Verification utilities
- `tools/check_model.py <seg>` — predictor vs sim accuracy.
- `tools/debug_mpc.py <seg> <controller> <step>` — inspect the optimized plan & predicted
  trajectory at one control step (this is how the off-manifold issue was found).
