# openpilot angle-steering tuning — investigation & handoff

> **Ready-to-apply:** `lat_smooth.patch` (verified: applies to `bold-keller`, valid Python,
> +23 lines, default no-op) + step-by-step in **`APPLY_TO_I6N.md`**. Harness: `replay_lat.py`.


Goal (requested): tune angle steering on `krvista/openpilot` (branch `i6n`, target car
**2026 Ioniq 6 N**) using the controls-challenge findings, validated by replaying the
`ccnc-drivelog` branch, and commit to `i6n`.

## Hard constraints hit in this session (read first)
1. **No write access to `krvista/openpilot`.** This session's git proxy authorizes only
   `krvista/controls_challenge` ("repository not authorized" for openpilot), and the GitHub
   MCP denies it. I can READ the fork (public) but **cannot push to `i6n`**. To have it
   committed there, add `krvista/openpilot` to the session's authorized repos (then I can
   commit), or apply the proposed change yourself.
2. **Replay fidelity.** I built an offline Python replay of the lateral *curvature* pipeline
   (no openpilot build needed — the rlogs already contain `modelV2`, `carState`,
   `liveParameters`, etc.). It parses the real drivelog fine, but does **not** reproduce the
   logged `actuators.curvature` exactly (full-pipeline corr ≈ 0.27; the stateful
   blend/clip + SubMaster timing aren't perfectly mirrored). So it is useful for **relative
   A/B** of a knob, **not** for an absolute drive-ready validation.
3. **Safety-critical.** Real-vehicle lateral control. Any change must stay conservative,
   behind a flag, respect openpilot's `clip_curvature` ISO jerk/accel limits, and be
   validated on-road. Nothing here is drive-ready as-is.

## What the fork already does (good news)
The controls-challenge idea is **already implemented** in `selfdrive/controls/controlsd.py`:
- `_lookahead_curvature()` — Phase-7 speed-adaptive path lookahead (phase lead).
- confidence-weighted curvature damping (yStd[5] + laneLineProbs) — jitter suppression.
- lane-departure protection, `LAT_SMOOTH_SECONDS`, roll compensation in `latcontrol_angle.py`.
The comments already cite this exact CCNC drivelog (route 0x05, "70 osc/min during MADS",
`yStd[5] p99=0.071`). So "tuning" here = adjusting these knobs, not adding the concept.

## Findings from the real drivelog (4 segments replayed)
- Logged commanded curvature ≈ the model's raw `action.desiredCurvature` (**corr 0.96**,
  RMS 0.014) — the pipeline largely passes the model command through.
- The oscillation is **temporal** (frame-to-frame model jitter, ~644 sign-changes/min in
  curvature-rate on moving frames), **not spatial** along the path.
- A **forward-window (spatial) smoothing** of the lookahead curvature (`smooth_win` knob I
  added to the harness) reduced the harness's own curvature-jerk only ~10% and saturated
  immediately — confirming the jitter isn't spatial.
- **Implication:** the effective lever is **temporal** smoothing of the curvature command.
  Because openpilot already has the *future* path, you can combine the existing phase-lead
  lookahead with a short temporal low-pass to cut jerk **without net lag** — the exact
  tracking/jerk tradeoff the controls-challenge work optimized (there: a forward-looking
  Gaussian window beat a causal EMA).

## Proposed change (conservative, FLAGGED, NOT yet drive-validated)
In `controlsd.state_control()`, after `new_desired_curvature` is computed and before
`clip_curvature`, optionally apply a small forward-lead temporal low-pass that uses the
model's near-future curvature so it does not add lag, e.g. blend the lookahead command with
a 1-2 frame prediction and a light EMA, gated by a param (default OFF). Keep `clip_curvature`
untouched (safety). Tune the EMA time-constant / lead on-device while watching the same
`osc/min` and lateral-jerk metrics the fork already logs.

This file set lets you A/B such knobs offline first:

## Files
- `parse_test.py` — proves offline rlog parsing (pycapnp + cereal schemas).
- `replay_lat.py` — reproduces the curvature pipeline; `DEFAULTS` holds the tunable knobs;
  `replay()`/`metrics()` give jerk_rms + osc/min for A/B.

## How to run
```
pip install pycapnp zstandard
# copy cereal schemas from the openpilot checkout:
#   cereal/{log,car(->opendbc),legacy,custom}.capnp + cereal/include/c++.capnp  -> ./cereal/
# fetch rlog.zst segments from the ccnc-drivelog branch into ./logs/
python replay_lat.py        # validation + logged-vs-recon metrics
```

## Honest bottom line
A faithful, drive-ready, replay-validated tune is **not** something I could responsibly
finalize in this session (no push access + imperfect replay fidelity + safety). What is
delivered: a working offline replay scaffold on the real Ioniq 6 N data, the signal analysis
pinpointing temporal (not spatial) jitter as the lever, and a conservative flagged change
direction consistent with the controls-challenge results — to be refined and validated
on-device, and committed to `i6n` once this session has write access to the openpilot repo.
