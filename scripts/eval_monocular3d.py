"""Phase 1: reproduce the single-image 3D ball localization baseline.

For each test frame with a ball annotation and calibration, estimate the 3D
ball position from that one view using the ball-size prior (range = f*D/d),
then measure the Euclidean error in meters against the triangulated ball_3D
ground truth.

This uses the ground-truth 2D annotation (no detector) so it isolates the
monocular geometry — the same quantity the paper reports as its single-image
baseline. We sweep the apparent-diameter source because the paper optimizes
the box diameter (optimized_d); comparing bbox-derived vs. optimized_d shows
how much the size estimate drives the error.

Usage:
    python scripts/eval_monocular3d.py --out eval/monocular3d_baseline.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from v3d import Camera
from v3d.geometry import D_REAL, apparent_diameter, localize_ball_monocular
from v3d.metrics import localization_error_stats
from v3d.snv3d import load_snv3d_csv, load_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DATA / "annotations" / "SNv3D.csv"))
    ap.add_argument("--split", default=str(DATA / "splits" / "SNv3D-test.txt"))
    ap.add_argument("--real-diameter", type=float, default=D_REAL)
    ap.add_argument("--out", default=str(ROOT / "eval" / "monocular3d_baseline.csv"))
    args = ap.parse_args()

    df = load_snv3d_csv(args.csv)
    test_keys = set(load_split(args.split))
    df = df[
        df["frame_id"].isin(test_keys)
        & df["ball_3D"].notna()
        & df["ball_bbox"].notna()
        & df["calibration"].notna()
    ].copy()

    # Diameter sources to compare.
    sources = {
        "bbox_mean": lambda r: apparent_diameter(r["ball_bbox"], "mean"),
        "bbox_max": lambda r: apparent_diameter(r["ball_bbox"], "max"),
    }
    if "optimized_d" in df.columns:
        sources["optimized_d"] = lambda r: (
            float(r["optimized_d"]) if pd.notna(r["optimized_d"]) else np.nan
        )

    records = []
    errs = {k: [] for k in sources}
    for _, r in df.iterrows():
        cam = Camera.from_soccernet(r["calibration"])
        bx, by, bw, bh = r["ball_bbox"]
        center = np.array([bx + bw / 2, by + bh / 2])
        gt = r["ball_3D"]
        rec = {"frame_id": r["frame_id"]}
        for name, fn in sources.items():
            d = fn(r)
            if not np.isfinite(d) or d <= 0:
                rec[f"err_{name}"] = np.nan
                continue
            est = localize_ball_monocular(cam, center, d, args.real_diameter)
            e = float(np.linalg.norm(est - gt))
            rec[f"err_{name}"] = e
            errs[name].append(e)
        records.append(rec)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out, index=False)

    print(f"monocular 3D ball localization on SNv3D-test  (n={len(records)} frames)")
    print(f"real ball diameter prior: {args.real_diameter} m\n")
    for name in sources:
        stats = localization_error_stats(np.array(errs[name]))
        if stats.get("n"):
            print(
                f"[{name:12s}] n={stats['n']:4d}  "
                f"mean={stats['mean_m']:.3f}  median={stats['median_m']:.3f}  "
                f"p90={stats['p90_m']:.3f}  rmse={stats['rmse_m']:.3f} m"
            )
    print(f"\nper-frame CSV -> {out}")


if __name__ == "__main__":
    main()
