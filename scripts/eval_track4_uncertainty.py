"""Track 4: does predicted triangulation uncertainty predict actual 3D error?

Real multi-view data here is only ever 2 views (SoccerNet-v3D has no >=3-view
groups; ISSIA has 9 frames with 3 cameras), so leave-one-out *across views* is
impossible. Instead we use leave-one-out **in time** as a held-out reference:

    for frame t, fit a quadratic to the triangulated positions in a window
    AROUND t but EXCLUDING t, predict the position at t, and take
    residual = || X_t - prediction ||.

The ball's true motion is smooth over ~a few frames, while triangulation error
is high-frequency, so this residual is a per-frame error proxy that never sees
frame t's own observation. We then ask whether the predicted sigma (and the
parallax angle) ranks those errors.

Usage:
    python scripts/eval_track4_uncertainty.py --window 7 --pixel-sigma 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from v3d import Camera
from v3d.issia import ISSIA_FPS, load_issia_calibration, load_issia_csv
from v3d.snv3d import load_snv3d_csv
from v3d.uncertainty import triangulate_with_covariance

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# A ball outside these bounds is physically impossible (pitch is 105x68 m,
# z is DOWN so z>1 is underground and z<-15 is 15 m in the air).
def implausible(x, y, z):
    return (z > 1.0) | (z < -15.0) | (np.abs(x) > 60.0) | (np.abs(y) > 45.0)


def soccernet_analysis(pixel_sigma: float, out_path: Path):
    """SoccerNet-v3D: quantify low-parallax uncertainty in the released annotations.

    Every released ball annotation comes from exactly 2 views (action + replay),
    and broadcast replays are often nearly collinear with the main camera — so
    this is where the paper's low-parallax warning actually bites.
    """
    df = load_snv3d_csv(DATA / "annotations" / "SNv3D.csv")
    r = df[df["ball_3D"].notna() & df["ball_bbox"].notna() & df["calibration"].notna()].copy()
    r["key3d"] = r["ball_3D"].map(lambda a: tuple(np.round(a, 4)))

    recs = []
    for (match, _), g in r.groupby(["match", "key3d"]):
        if len(g) < 2:
            continue
        cams = [Camera.from_soccernet(c) for c in g["calibration"]]
        pix = np.array([[b[0] + b[2] / 2, b[1] + b[3] / 2] for b in g["ball_bbox"]])
        try:
            res = triangulate_with_covariance(cams, pix, pixel_sigma=pixel_sigma)
        except Exception:
            continue
        recs.append(
            {
                "match": match, "n_views": res.n_views,
                "x": res.point[0], "y": res.point[1], "z": res.point[2],
                "sigma_m": res.sigma_m, "sigma_major_m": res.sigma_major_m,
                "parallax_deg": res.parallax_deg, "rms_reproj_px": res.rms_reproj_px,
            }
        )
    d = pd.DataFrame(recs)
    d["implausible"] = implausible(d.x, d.y, d.z)
    d.to_csv(out_path, index=False)

    print("=" * 70)
    print("SoccerNet-v3D — uncertainty of the released 2-view ball annotations")
    print("=" * 70)
    print(f"groups: {len(d)}")
    print(f"parallax (deg): median {d.parallax_deg.median():.1f}  "
          f"p5 {d.parallax_deg.quantile(.05):.1f}  min {d.parallax_deg.min():.2f}")
    print(f"  below 10deg: {(d.parallax_deg<10).mean():.1%}   below 5deg: {(d.parallax_deg<5).mean():.1%}")
    print(f"predicted sigma (m): median {d.sigma_m.median():.3f}  p99 {d.sigma_m.quantile(.99):.2f}  "
          f"max {d.sigma_m.max():.1f}")
    print(f"  sigma>0.5m: {(d.sigma_m>0.5).mean():.1%}   >1m: {(d.sigma_m>1).mean():.1%}   "
          f">5m: {(d.sigma_m>5).mean():.1%}")

    hi, lo = d[d.sigma_m > 1], d[d.sigma_m <= 1]
    sp = d[["sigma_m", "rms_reproj_px"]].corr(method="spearman").iloc[0, 1]
    print("\nTHE TRAP (reprojection error cannot see this):")
    print(f"  median reprojection err, sigma>1m : {hi.rms_reproj_px.median():.2f} px")
    print(f"  median reprojection err, sigma<=1m: {lo.rms_reproj_px.median():.2f} px")
    print(f"  spearman(sigma, reproj) = {sp:+.3f}  -> uncertain points look BETTER by reprojection")

    print("\npredicted sigma vs physically-impossible positions (real-data validation):")
    bins = [(0, .1), (.1, .5), (.5, 1), (1, 5), (5, np.inf)]
    for a, b in bins:
        s = d[(d.sigma_m >= a) & (d.sigma_m < b)]
        if len(s) >= 5:
            print(f"  sigma[{a},{b}): n={len(s):4d}  implausible={s.implausible.mean():6.1%}  "
                  f"median parallax={s.parallax_deg.median():5.1f}deg")

    print("\ngating on parallax (reject ill-conditioned pairs):")
    base = d.implausible.mean()
    for thr in (0, 5, 10, 15, 20):
        k = d[d.parallax_deg >= thr]
        print(f"  parallax>={thr:2d}deg: keep {len(k)/len(d):5.1%}  implausible={k.implausible.mean():6.2%}"
              f"  ({100*(1-k.implausible.mean()/max(base,1e-9)):+.0f}% vs no gate)")
    print(f"\nper-group CSV -> {out_path}")
    return d


def build_observations(df, cams):
    """Per-frame list of (camera_indices, pixels) for frames with >= 2 views."""
    rows = []
    for _, r in df.iterrows():
        idx, pix = [], []
        for c in range(1, 7):
            x, y = r.get(f"x_cam{c}"), r.get(f"y_cam{c}")
            if pd.notna(x) and pd.notna(y):
                idx.append(c)
                pix.append([float(x), float(y)])
        if len(idx) >= 2:
            rows.append({"frame": int(r["frame"]), "cams": idx, "pix": np.array(pix)})
    return rows


def loo_temporal_residual(frames, points, window, min_neighbors=4):
    """Leave-one-out temporal prediction residual per frame (meters)."""
    frames = np.asarray(frames)
    pts = np.asarray(points)
    half = window // 2
    out = np.full(len(frames), np.nan)
    for i, f in enumerate(frames):
        sel = (frames >= f - half) & (frames <= f + half) & (frames != f)
        if sel.sum() < min_neighbors:
            continue
        t = (frames[sel] - f) * (1.0 / ISSIA_FPS)
        # Quadratic fit per axis on neighbors only; predict at t=0.
        A = np.vstack([np.ones_like(t), t, t**2]).T
        try:
            coef, *_ = np.linalg.lstsq(A, pts[sel], rcond=None)
        except np.linalg.LinAlgError:
            continue
        pred = coef[0]  # value at t=0
        out[i] = float(np.linalg.norm(pts[i] - pred))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=7, help="LOO temporal window (frames)")
    ap.add_argument("--pixel-sigma", type=float, default=1.0)
    ap.add_argument("--dataset", choices=["both", "issia", "soccernet"], default="both")
    ap.add_argument("--out", default=str(ROOT / "eval" / "track4_uncertainty.csv"))
    args = ap.parse_args()

    if args.dataset in ("both", "soccernet"):
        soccernet_analysis(args.pixel_sigma,
                           Path(args.out).with_name("track4_soccernet.csv"))
        if args.dataset == "soccernet":
            return
        print()

    df = load_issia_csv(DATA / "annotations" / "ISSIA-3D.csv")
    cams = load_issia_calibration(DATA / "annotations" / "issia_calibration.json")
    obs = build_observations(df, cams)

    recs = []
    for o in obs:
        cl = [cams[c] for c in o["cams"]]
        try:
            r = triangulate_with_covariance(cl, o["pix"], pixel_sigma=args.pixel_sigma)
        except Exception:
            continue
        recs.append(
            {
                "frame": o["frame"], "n_views": r.n_views,
                "x": r.point[0], "y": r.point[1], "z": r.point[2],
                "sigma_m": r.sigma_m, "sigma_major_m": r.sigma_major_m,
                "parallax_deg": r.parallax_deg, "rms_reproj_px": r.rms_reproj_px,
            }
        )
    res = pd.DataFrame(recs).sort_values("frame").reset_index(drop=True)

    res["loo_resid_m"] = loo_temporal_residual(
        res["frame"].to_numpy(), res[["x", "y", "z"]].to_numpy(), args.window
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)

    v = res[res["loo_resid_m"].notna() & np.isfinite(res["sigma_m"])].copy()
    print("=" * 70)
    print(f"ISSIA-3D — uncertainty vs held-out temporal error (pixel_sigma="
          f"{args.pixel_sigma} px, LOO window={args.window})")
    print("=" * 70)
    print(f"frames triangulated: {len(res)}   with LOO reference: {len(v)}\n")

    print(f"parallax:  median {v.parallax_deg.median():.1f}deg  "
          f"p10 {v.parallax_deg.quantile(.1):.1f}  p90 {v.parallax_deg.quantile(.9):.1f}")
    print(f"predicted sigma: median {v.sigma_m.median():.2f} m  "
          f"p90 {v.sigma_m.quantile(.9):.2f} m")
    print(f"LOO residual:    median {v.loo_resid_m.median():.2f} m\n")

    # Does predicted sigma rank actual error?
    sp = v[["sigma_m", "loo_resid_m"]].corr(method="spearman").iloc[0, 1]
    sp_par = v[["parallax_deg", "loo_resid_m"]].corr(method="spearman").iloc[0, 1]
    sp_rep = v[["rms_reproj_px", "loo_resid_m"]].corr(method="spearman").iloc[0, 1]
    print("Spearman rank correlation with actual (LOO) error:")
    print(f"  predicted sigma   {sp:+.3f}   <- should be strongly POSITIVE")
    print(f"  parallax angle    {sp_par:+.3f}   <- should be NEGATIVE (more parallax, less error)")
    print(f"  reprojection err  {sp_rep:+.3f}   <- the paper's point: weak predictor\n")

    print("error by predicted-sigma decile (monotonic => sigma is informative):")
    v["dec"] = pd.qcut(v["sigma_m"], 10, labels=False, duplicates="drop")
    g = v.groupby("dec").agg(n=("loo_resid_m", "size"),
                             sigma=("sigma_m", "median"),
                             parallax=("parallax_deg", "median"),
                             err=("loo_resid_m", "median"))
    print(g.round(2).to_string())

    print("\nrejecting the least-certain windows (gate on predicted sigma):")
    base = v["loo_resid_m"].median()
    for keep in (1.0, 0.9, 0.75, 0.5):
        thr = v["sigma_m"].quantile(keep)
        k = v[v["sigma_m"] <= thr]
        print(f"  keep best {keep:4.0%} by sigma: n={len(k):4d}  median err={k.loo_resid_m.median():.2f} m"
              f"  ({100*(1-k.loo_resid_m.median()/base):+.0f}% vs keeping all)")

    print(f"\nper-frame CSV -> {args.out}")


if __name__ == "__main__":
    main()
