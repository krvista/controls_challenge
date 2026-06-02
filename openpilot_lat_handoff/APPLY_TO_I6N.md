# Apply + validate: angle-steering lead-smoothing on `i6n` (2026 Ioniq 6 N)

Actionable guide for a Claude Code session that **has write access to `krvista/openpilot`**.
Goal: land a conservative, replay-validated angle-steering tweak on branch `i6n`
(base `claude/bold-keller-S8dqp`), validated against the `ccnc-drivelog` rlogs.

> Prereq that blocked the original session: the git proxy here only authorizes
> `controls_challenge`. Run this from a session whose environment includes
> `krvista/openpilot` with Contents:Write.

## The change — `lat_smooth.patch`
One file: `selfdrive/controls/controlsd.py` (+23 lines, **defaults = byte-for-byte no-op**).
It adds a **forward-looking (lead) low-pass on the commanded curvature** — the controls-
challenge result mapped to openpilot: a temporal low-pass cuts frame-to-frame jerk, and the
extra lookahead time cancels its lag (openpilot already knows the future path), so net phase
stays ~0. Two module constants:
- `LAT_CMD_SMOOTH_TAU` (s) — low-pass time constant on `new_desired_curvature`.
- `LAT_CMD_LOOKAHEAD_EXTRA_S` (s) — extra lookahead lead to compensate the low-pass lag.

It is applied **after** confidence damping / lane-departure and **before** `clip_curvature`,
so openpilot's ISO lateral jerk/accel limits still bound the output. `clip_curvature` is
untouched (safety preserved).

## Steps
```bash
# 1. base + branch
git fetch origin claude/bold-keller-S8dqp i6n
git switch -c i6n-work origin/i6n            # or start from bold-keller if i6n should track it
# 2. apply (patch is in the controls_challenge repo: openpilot_lat_handoff/lat_smooth.patch)
git apply --3way /path/to/lat_smooth.patch
python -m py_compile selfdrive/controls/controlsd.py
# 3. pick starting tune (CONSERVATIVE) — edit the two constants in controlsd.py:
#    LAT_CMD_SMOOTH_TAU = 0.06 ; LAT_CMD_LOOKAHEAD_EXTRA_S = 0.06
```

## Validate (do this BEFORE committing aggressive values)

### A) Quick offline A/B (no openpilot build) — `replay_lat.py`
Relative jerk/oscillation comparison of the curvature pipeline on the real drivelog.
```bash
pip install pycapnp zstandard
# cereal schemas from this checkout -> ./cereal/ :
#   cereal/log.capnp, cereal/legacy.capnp, cereal/custom.capnp, cereal/include/c++.capnp
#   cereal/car.capnp  (symlink -> use opendbc_repo/opendbc/car/car.capnp contents)
# rlogs from ccnc-drivelog branch -> ./logs/seg*.rlog.zst  (git show HEAD:drivelog/<f> > ...)
python replay_lat.py                 # prints jerk_rms + osc/min for the pipeline
```
A/B the knobs by setting `DEFAULTS['smooth_win']` / adding a temporal-EMA mirror of the patch
in `replay_lat.py` and comparing `metrics()` (jerk_rms, osc/min) and `RMS(cmd-model)`
(tracking proxy — must stay ~unchanged) and the cmd-vs-model cross-correlation lag peak
(must stay ~0 → confirms no net lag).

> Fidelity caveat: this standalone harness reproduces the *single-frame* lookahead well
> (corr ~0.86–0.93 vs model) but the **full stateful pipeline diverges from the logged
> curvature (corr ~0.27)** — SubMaster timing + stateful blend/clip aren't perfectly
> mirrored. Use it for **relative** A/B only. For exact validation use (B).

### B) Gold standard — openpilot native replay (exact, deterministic)
Use openpilot's own replay to compare commanded curvature/angle before vs after on the route:
```bash
# in the openpilot checkout (built env):
#   tools/replay or selfdrive/test/process_replay on the ccnc-drivelog route,
#   diff carControl.actuators.{curvature,steeringAngleDeg} old vs patched.
```
Accept only if: lateral **jerk / osc-per-min ↓**, tracking of `modelV2.action.desiredCurvature`
**unchanged**, no new `clip_curvature` (curvatureLimited) trips, and lag ≈ 0.

## Tuning guidance
- Start `TAU = 0.06`, `EXTRA ≈ TAU` (0.06). Sweep TAU 0.04–0.10.
- Keep `EXTRA ≈ TAU` so the lead cancels the low-pass lag (verify lag≈0 in replay).
- The fork already logs osc/min metrics it was tuned with — watch those.

## Safety (must)
- Real 2026 Ioniq 6 N lateral control. **On-road validation required** before relying on it.
- Ship with conservative values; keep an instant disable (`LAT_CMD_SMOOTH_TAU = 0.0`).
- `clip_curvature` (ISO jerk/accel/max-curvature) is intentionally left intact and downstream.
- This package's offline numbers are **relative** indicators, not a road sign-off.

## Commit
```bash
git commit -am "lat: forward-lead low-pass on commanded curvature (jerk reduction, default off)"
git push origin HEAD:i6n
```
Then report before/after replay metrics in the PR/commit body, noting on-road validation TODO.
