"""Build a feedforward inverse model: steer = g(lataccel, roll_lataccel, v_ego, a_ego).

Two data sources:
  --source data  : fit from the real driving CSVs (steerCommand vs achieved target_lataccel).
                   Fast, no ONNX. Great prior since the sim is trained to mimic this.
  --source sim   : on-manifold probing of the ONNX plant (steady-state steer->lataccel).

We model the steady-state relation as a speed-scheduled affine map:
    steer ~= a(v) * (lataccel - roll_lataccel) + c(v)
which inverts to the controller's feedforward. We fit per v_ego bin and also expose
a smooth fallback. Saved as an .npz lookup consumed by controllers/ff_pid.py.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "data"
ACC_G = 9.81

V_BINS = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 100], dtype=float)


def load_real(n_segs=2000, stride=1):
    """Collect (v_ego, roll_lataccel, a_ego, lataccel, steer) rows from real CSVs."""
    rows_v, rows_roll, rows_a, rows_lat, rows_steer = [], [], [], [], []
    files = sorted(DATA_DIR.iterdir())[:n_segs]
    for f in files:
        df = pd.read_csv(f)
        roll = np.sin(df["roll"].values) * ACC_G
        v = df["vEgo"].values
        a = df["aEgo"].values
        lat = df["targetLateralAcceleration"].values
        steer = -df["steerCommand"].values  # sim convention (right-positive)
        rows_v.append(v[::stride]); rows_roll.append(roll[::stride])
        rows_a.append(a[::stride]); rows_lat.append(lat[::stride])
        rows_steer.append(steer[::stride])
    return (np.concatenate(rows_v), np.concatenate(rows_roll),
            np.concatenate(rows_a), np.concatenate(rows_lat), np.concatenate(rows_steer))


def fit_speed_scheduled(v, roll, a, lat, steer):
    """Per v-bin least squares: steer = a_v*(lat - roll) + b_v*a_ego + c_v."""
    eff = lat - roll  # lataccel to be produced by steering (roll compensated)
    finite = np.isfinite(v) & np.isfinite(roll) & np.isfinite(a) & np.isfinite(lat) & np.isfinite(steer)
    centers, A, B, C = [], [], [], []
    for lo, hi in zip(V_BINS[:-1], V_BINS[1:]):
        m = (v >= lo) & (v < hi) & finite
        if m.sum() < 200:
            continue
        X = np.column_stack([eff[m], a[m], np.ones(m.sum())])
        coef, *_ = np.linalg.lstsq(X, steer[m], rcond=None)
        centers.append((lo + min(hi, 45)) / 2)
        A.append(coef[0]); B.append(coef[1]); C.append(coef[2])
        pred = X @ coef
        r2 = 1 - np.sum((steer[m] - pred) ** 2) / np.sum((steer[m] - steer[m].mean()) ** 2)
        print(f"  v[{lo:.0f},{hi:.0f}) n={m.sum():>7d}  a_v={coef[0]:+.4f} b_v={coef[1]:+.4f} c_v={coef[2]:+.4f}  R2={r2:.3f}")
    return np.array(centers), np.array(A), np.array(B), np.array(C)


def fit_rich(v, roll, a, lat, steer):
    """Per v-bin LS with separate roll coef + mild saturation term:
        steer = a_v*lat + r_v*roll + q_v*lat*|lat| + b_v*a + c_v
    Stored as coef matrix (n_bins x 5) with columns [lat, roll, lat|lat|, a, 1]."""
    finite = np.isfinite(v) & np.isfinite(roll) & np.isfinite(a) & np.isfinite(lat) & np.isfinite(steer)
    centers, coefs = [], []
    for lo, hi in zip(V_BINS[:-1], V_BINS[1:]):
        m = (v >= lo) & (v < hi) & finite
        if m.sum() < 200:
            continue
        X = np.column_stack([lat[m], roll[m], lat[m] * np.abs(lat[m]), a[m], np.ones(m.sum())])
        coef, *_ = np.linalg.lstsq(X, steer[m], rcond=None)
        pred = X @ coef
        r2 = 1 - np.sum((steer[m] - pred) ** 2) / np.sum((steer[m] - steer[m].mean()) ** 2)
        centers.append((lo + min(hi, 45)) / 2)
        coefs.append(coef)
        print(f"  v[{lo:.0f},{hi:.0f}) n={m.sum():>7d}  lat={coef[0]:+.3f} roll={coef[1]:+.3f} "
              f"sat={coef[2]:+.4f} a={coef[3]:+.4f} c={coef[4]:+.4f}  R2={r2:.3f}")
    return np.array(centers), np.array(coefs)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="data", choices=["data", "sim"])
    p.add_argument("--model", default="affine", choices=["affine", "rich"])
    p.add_argument("--n_segs", type=int, default=2000)
    p.add_argument("--out", default="")
    args = p.parse_args()

    print(f"Loading {args.n_segs} real segments...")
    v, roll, a, lat, steer = load_real(args.n_segs)
    print(f"rows: {len(v)}")
    if args.model == "affine":
        out = args.out or str(ROOT / "controllers" / "ff_model.npz")
        print("Fitting speed-scheduled affine inverse model:")
        centers, A, B, C = fit_speed_scheduled(v, roll, a, lat, steer)
        np.savez(out, v_centers=centers, a_v=A, b_v=B, c_v=C)
    else:
        out = args.out or str(ROOT / "controllers" / "ff_model_v2.npz")
        print("Fitting rich (separate roll + saturation) inverse model:")
        centers, coefs = fit_rich(v, roll, a, lat, steer)
        np.savez(out, v_centers=centers, coefs=coefs)
    print(f"Saved -> {out}")
