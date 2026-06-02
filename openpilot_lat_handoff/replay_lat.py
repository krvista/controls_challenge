"""Offline replay of openpilot's lateral CURVATURE pipeline on real rlogs.

Reproduces controlsd's per-cycle desired_curvature computation
(_lookahead_curvature -> confidence damping -> lane-departure block -> clip_curvature)
from logged modelV2 / carState / liveParameters / driverAssistance, and compares to
the logged carControl.actuators.curvature to validate, then A/B tests tuning knobs.

Works in curvature space (monotone proxy for the commanded steering angle), so it needs
no VehicleModel build and no openpilot runtime.
"""
import sys, glob, json
import numpy as np
import zstandard as zstd
import capnp
capnp.remove_import_hook()

CER = "/tmp/replay/cereal"
log_capnp = capnp.load(f"{CER}/log.capnp", imports=[CER, "/tmp/replay"])

DT_CTRL = 0.01
ACC_G = 9.80665
MAX_LATERAL_JERK = 5.0
MAX_LATERAL_ACCEL_NO_ROLL = 3.0
MAX_CURVATURE = 0.2
MIN_SPEED = 1.0

# ---- default (stock-fork) lookahead knobs ----
DEFAULTS = dict(
    base_v=[5.6, 13.9, 27.8], base_s=[0.08, 0.10, 0.13],
    boost_c=[0.001, 0.005], boost_s=[0.0, 0.12], t_cap=0.25, dist_cap=10.0,
    conf_ystd=[0.05, 0.30], conf_lane=[0.05, 0.30],
    smooth_win=0,   # NEW knob: 0 = single-point (stock); >0 = average curvature over
                    # this many forward path samples around the lookahead point (jerk-opt)
)


def clip_curvature(v_ego, prev, new, roll):
    v_ego = max(v_ego, MIN_SPEED)
    rate = MAX_LATERAL_JERK / (v_ego ** 2)
    new = np.clip(new, prev - rate * DT_CTRL, prev + rate * DT_CTRL)
    rollc = roll * ACC_G
    new = np.clip(new, (-MAX_LATERAL_ACCEL_NO_ROLL + rollc) / v_ego**2,
                       (MAX_LATERAL_ACCEL_NO_ROLL + rollc) / v_ego**2)
    return float(np.clip(new, -MAX_CURVATURE, MAX_CURVATURE))


def lookahead_curvature(px, py, fallback, v_ego, K):
    n = len(px)
    if n < 5:
        return fallback
    abs_curv = abs(fallback)
    if abs_curv < 0.001:
        return fallback
    base_s = float(np.interp(v_ego, K['base_v'], K['base_s']))
    boost_s = float(np.interp(abs_curv, K['boost_c'], K['boost_s']))
    t_ahead = min(base_s + boost_s, K['t_cap'])
    dist_ahead = min(v_ego * t_ahead, K['dist_cap'])
    if dist_ahead < 0.3:
        return fallback
    n = min(n, 12)
    x = px[:n]; y = py[:n]
    if x[-1] < dist_ahead or not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return fallback
    try:
        c = np.polyfit(x, y, 3)
    except (np.linalg.LinAlgError, ValueError):
        return fallback
    win = K['smooth_win']
    if win and win > 0:
        # jerk-optimal: average curvature over a small forward window around dist_ahead
        ds = np.linspace(max(0.3, dist_ahead - 1.0), dist_ahead + 1.0, win)
        curv = np.mean(6.0 * c[0] * ds + 2.0 * c[1])
    else:
        curv = 6.0 * c[0] * dist_ahead + 2.0 * c[1]
    return float(curv) if np.isfinite(curv) else fallback


def events(path):
    with open(path, "rb") as f:
        data = zstd.ZstdDecompressor().decompress(f.read(), max_output_size=400_000_000)
    return list(log_capnp.Event.read_multiple_bytes(data))


def replay(paths, K, validate=False):
    cur = {}            # latest message snapshots
    desired = 0.0
    rec, logged, vs, model_dc = [], [], [], []
    for path in paths:
        evs = events(path)
        evs.sort(key=lambda e: e.logMonoTime)
        desired = 0.0
        for ev in evs:
            w = ev.which()
            if w in ("carState", "modelV2", "liveParameters", "driverAssistance"):
                cur[w] = ev
            elif w == "carControl":
                cc = ev.carControl
                cs = cur.get("carState"); m = cur.get("modelV2"); lp = cur.get("liveParameters")
                if cs is None or m is None or lp is None:
                    continue
                cs = cs.carState; m = m.modelV2; lp = lp.liveParameters
                v_ego = cs.vEgo; roll = lp.roll
                px = np.array(m.position.x, dtype=np.float64)
                py = np.array(m.position.y, dtype=np.float64)
                try:
                    fb = m.action.desiredCurvature
                except Exception:
                    fb = 0.0
                # Lateral was operating via MADS in this log (cc.latActive logs False but
                # actuators.curvature tracks the model). Run the active pipeline when moving.
                active = v_ego > MIN_SPEED
                if not active:
                    new = desired
                else:
                    new = lookahead_curvature(px, py, fb, v_ego, K)
                    # confidence damping (angle cars)
                    ystd = float(m.position.yStd[5]) if len(m.position.yStd) > 5 else 0.0
                    lp_probs = m.laneLineProbs
                    lane_min = min(float(lp_probs[1]), float(lp_probs[2])) if len(lp_probs) >= 4 else 1.0
                    conf_y = float(np.interp(ystd, K['conf_ystd'], [1.0, 0.0]))
                    conf_l = float(np.interp(lane_min, K['conf_lane'], [0.0, 1.0]))
                    conf = min(conf_y, conf_l)
                    if conf < 1.0:
                        new = conf * new + (1.0 - conf) * desired
                    # lane-departure block
                    blinker = bool(cs.leftBlinker or cs.rightBlinker)
                    da_ev = cur.get("driverAssistance")
                    if not blinker and da_ev is not None:
                        da = da_ev.driverAssistance
                        if (da.leftLaneDeparture and new < desired) or (da.rightLaneDeparture and new > desired):
                            new = desired
                desired = clip_curvature(v_ego, desired, new, roll)
                rec.append(desired); logged.append(cc.actuators.curvature)
                vs.append(v_ego); model_dc.append(fb)
    rec = np.array(rec); logged = np.array(logged); vs = np.array(vs); model_dc = np.array(model_dc)
    return rec, logged, vs, model_dc


def metrics(curv, vs, dt=DT_CTRL):
    moving = vs > 3.0
    c = curv[moving]
    if len(c) < 10:
        return {}
    jerk_rms = float(np.sqrt(np.mean((np.diff(c) / dt) ** 2)))  # curvature-rate proxy for jerk
    # oscillations: sign changes of curvature-rate per minute
    dc = np.diff(c)
    sign_changes = int(np.sum(np.diff(np.sign(dc)) != 0))
    osc_per_min = sign_changes / (len(c) * dt) * 60.0
    return dict(n=int(len(c)), jerk_rms=round(jerk_rms, 5), osc_per_min=round(osc_per_min, 1))


if __name__ == "__main__":
    paths = sorted(glob.glob("/tmp/replay/logs/seg*.rlog.zst"))
    print(f"segments: {len(paths)}")
    rec, logged, vs, mdc = replay(paths, DEFAULTS, validate=True)
    moving = vs > 3.0
    rms = float(np.sqrt(np.mean((rec[moving] - logged[moving]) ** 2)))
    corr = float(np.corrcoef(rec[moving], logged[moving])[0, 1])
    print(f"VALIDATION (recon OLD vs logged curvature, moving): RMS={rms:.6f} corr={corr:.4f} n={moving.sum()}")
    print(f"  logged metrics : {metrics(logged, vs)}")
    print(f"  recon  metrics : {metrics(rec, vs)}")
