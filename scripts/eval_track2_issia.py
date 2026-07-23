"""Track 2: physics-constrained monocular ball depth on ISSIA-3D.

Compares three single-camera 3D localization strategies against the
triangulated `ball_3D` ground truth, on continuous 25 fps ISSIA footage:

  baseline   per-frame monocular size prior (range = f*D/opt_d)  [Phase 1 method]
  physics    ballistic fit over a temporal window of 2D points   [Track 2]
  (the multi-view triangulated ball_3D is the GT, i.e. the practical upper bound)

For each camera track and each center frame with GT, a window of +/- W/2 frames
(same camera, no frame gap larger than `max_gap`) is fit with a gravity-
constrained trajectory; the estimate at the center frame is scored. We report
overall and split by AIRBORNE (|z_gt| above `air_thresh` m) vs near-ground,
since gravity is only informative when the ball accelerates vertically.

Usage:
    python scripts/eval_track2_issia.py --window 9 --cameras 3 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from v3d.geometry import localize_ball_monocular
from v3d.issia import ISSIA_FPS, issia_camera_track, load_issia_calibration, load_issia_csv
from v3d.metrics import localization_error_stats
from v3d.trajectory import GRAVITY, fit_ballistic_single_view

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def parse_xyz(v):
    if v is None:
        return None
    return np.asarray(v, dtype=float).reshape(3)


def eval_camera(track: pd.DataFrame, cam, window: int, max_gap: int,
                dt: float, real_d: float, prior_pos: float, prior_vel: float):
    half = window // 2
    frames = track["frame"].to_numpy()
    recs = []
    for i in range(len(track)):
        gt = parse_xyz(track["ball_3D"].iloc[i])
        if gt is None:
            continue
        f0 = frames[i]
        # Window: same-camera observations within +/- half frames, contiguous.
        lo, hi = f0 - half, f0 + half
        w = track[(track["frame"] >= lo) & (track["frame"] <= hi)]
        if len(w) < 3:
            continue
        wf = w["frame"].to_numpy()
        if np.max(np.diff(wf)) > max_gap:
            continue

        times = (wf - f0) * dt
        pix = w[["u", "v"]].to_numpy()
        wd = w["opt_d"].to_numpy()

        # Per-frame size-prior 3D points across the window -> stable init.
        init_pts = []
        for (u, v), dd in zip(pix, wd):
            if np.isfinite(dd) and dd > 0:
                init_pts.append(localize_ball_monocular(cam, np.array([u, v]), float(dd), real_d))
            else:
                init_pts.append(None)
        have_all = all(p is not None for p in init_pts)

        # Physics fit; estimate at center (t=0 -> X0).
        try:
            tr = fit_ballistic_single_view(
                cam, times, pix, GRAVITY, refine=True,
                init_points=np.array(init_pts) if have_all else None,
                prior_pos_px_per_m=prior_pos, prior_vel_px_per_mps=prior_vel,
            )
            est_phys = tr.X0
            err_phys = float(np.linalg.norm(est_phys - gt))
            cond = tr.condition
            rms = tr.rms_reproj_px
            speed = float(np.linalg.norm(tr.V0))
            vspeed = float(abs(tr.V0[2]))
        except Exception:
            err_phys, cond, rms, speed, vspeed = (np.nan,) * 5

        # How much the ball actually moves in-image across the window: if it
        # barely moves, the window carries no depth information and the fit
        # just returns the (biased) size-prior init.
        pix_disp = float(np.linalg.norm(pix.max(axis=0) - pix.min(axis=0)))

        # Baseline: size prior at center frame.
        d = track["opt_d"].iloc[i]
        err_base = np.nan
        if np.isfinite(d) and d > 0:
            center = np.array([track["u"].iloc[i], track["v"].iloc[i]])
            est_base = localize_ball_monocular(cam, center, float(d), real_d)
            err_base = float(np.linalg.norm(est_base - gt))

        recs.append(
            {
                "frame": int(f0), "n_obs": len(w),
                "z_gt": gt[2], "airborne": abs(gt[2]) > 0.0,  # filled below
                "err_phys": err_phys, "err_base": err_base,
                "cond": cond, "rms_reproj": rms,
                "speed_mps": speed, "vspeed_mps": vspeed, "pix_disp": pix_disp,
            }
        )
    return recs


def summarize(df, col, mask, label):
    e = df.loc[mask, col].to_numpy()
    s = localization_error_stats(e)
    if not s.get("n"):
        return f"  {label:34s} (no samples)"
    return (f"  {label:34s} n={s['n']:4d}  MAEm={s['mean_m']:5.2f}  "
            f"median={s['median_m']:5.2f}  P2m={s['p2m']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=9, help="temporal window (frames)")
    ap.add_argument("--max-gap", type=int, default=2, help="max frame gap inside window")
    ap.add_argument("--cameras", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--air-thresh", type=float, default=1.0, help="|z|>thr => airborne (m)")
    ap.add_argument("--real-diameter", type=float, default=0.22)
    ap.add_argument("--prior-pos", type=float, default=0.3,
                    help="soft prior toward size-prior init, px per meter "
                         "(pins the depth null-direction for a single view; 0 = off)")
    ap.add_argument("--prior-vel", type=float, default=0.1,
                    help="soft prior on velocity, px per (m/s)")
    ap.add_argument("--gate-disp", type=float, default=25.0,
                    help="min in-image ball displacement (px) across the window; "
                         "below this the window carries no depth information")
    ap.add_argument("--gate-rms", type=float, default=3.0,
                    help="max reprojection RMS (px) for an accepted physics fit")
    ap.add_argument("--cond-max", type=float, default=1e5,
                    help="drop physics fits with condition number above this")
    ap.add_argument("--out", default=str(ROOT / "eval" / "track2_issia.csv"))
    args = ap.parse_args()

    df = load_issia_csv(DATA / "annotations" / "ISSIA-3D.csv")
    cams = load_issia_calibration(DATA / "annotations" / "issia_calibration.json")
    dt = 1.0 / ISSIA_FPS

    all_recs = []
    for c in args.cameras:
        track = issia_camera_track(df, c)
        recs = eval_camera(track, cams[c], args.window, args.max_gap, dt,
                           args.real_diameter, args.prior_pos, args.prior_vel)
        for r in recs:
            r["camera"] = c
        all_recs.extend(recs)

    res = pd.DataFrame(all_recs)
    res["airborne"] = res["z_gt"].abs() > args.air_thresh
    # Physics estimates that are ill-conditioned are unreliable -> mark.
    res["phys_ok"] = res["cond"] <= args.cond_max

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)

    print(f"Track 2 — ISSIA physics-constrained depth  (window={args.window} frames "
          f"= {args.window*dt*1000:.0f} ms, cameras={args.cameras})")
    print(f"windows evaluated: {len(res)}  |  airborne (|z|>{args.air_thresh}m): "
          f"{int(res['airborne'].sum())}\n")

    both = res["err_phys"].notna() & res["err_base"].notna()
    ok = both & res["phys_ok"]
    print("ALL windows (physics well-conditioned):")
    print(summarize(res, "err_base", ok, "baseline (size prior)"))
    print(summarize(res, "err_phys", ok, "physics (ballistic window)"))
    print("\nAIRBORNE windows only (gravity informative):")
    air = ok & res["airborne"]
    print(summarize(res, "err_base", air, "baseline (size prior)"))
    print(summarize(res, "err_phys", air, "physics (ballistic window)"))
    print("\nNEAR-GROUND windows only:")
    gnd = ok & ~res["airborne"]
    print(summarize(res, "err_base", gnd, "baseline (size prior)"))
    print(summarize(res, "err_phys", gnd, "physics (ballistic window)"))

    # Headline: gate out windows that carry no depth information (ball nearly
    # static in-image) or whose fit does not explain the observations.
    gate = ok & (res["pix_disp"] >= args.gate_disp) & (res["rms_reproj"] <= args.gate_rms)
    print(f"\nGATED (pix_disp>={args.gate_disp:.0f}px, rms<={args.gate_rms:.0f}px) "
          f"— {int(gate.sum())}/{int(ok.sum())} windows ({gate.sum()/max(ok.sum(),1):.0%}):")
    print(summarize(res, "err_base", gate, "baseline (size prior)"))
    print(summarize(res, "err_phys", gate, "physics (ballistic window)"))
    mb, mp = res.loc[gate, "err_base"].median(), res.loc[gate, "err_phys"].median()
    p2b = (res.loc[gate, "err_base"] <= 2).mean()
    p2p = (res.loc[gate, "err_phys"] <= 2).mean()
    if np.isfinite(mb) and mb > 0:
        print(f"  -> median {100*(1-mp/mb):+.0f}% ; P2m {p2b:.2f} -> {p2p:.2f} "
              f"({p2p/max(p2b,1e-9):.1f}x more estimates within 2 m)")
    print(f"\nper-window CSV -> {args.out}")


if __name__ == "__main__":
    main()
