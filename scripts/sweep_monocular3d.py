"""Part A: sweep the monocular-3D hyperparameters to characterize the MAEm gap.

The detector runs ONCE per frame (all boxes cached to a parquet), then we sweep
localization hyperparameters analytically over the cache:

  * confidence threshold  {0.25, 0.40, 0.50}
  * ball diameter prior    {0.21, 0.22, 0.23} m
  * apparent-diameter mode {mean, max, geom}

For every config we report the error over ALL fired frames and, separately,
over frames where the top detection actually localizes the ball (IoU with the
GT ball box >= 0.5). The paper's MAEm almost certainly reflects the latter,
since a detector misfire is a detection failure, not a 3D-localization error.

Outputs:
  eval/monocular3d_detcache.parquet   per-frame detections + geometry (cache)
  eval/monocular3d_sweep.csv          the full sweep grid

Usage:
    python scripts/sweep_monocular3d.py            # build cache if missing, sweep
    python scripts/sweep_monocular3d.py --rebuild  # force re-detection
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from v3d import Camera
from v3d.geometry import apparent_diameter, localize_ball_monocular
from v3d.metrics import iou_xywh, localization_error_stats
from v3d.snv3d import load_snv3d_csv, load_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / "eval" / "monocular3d_detcache.parquet"


def xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return np.array([x1, y1, x2 - x1, y2 - y1])


def build_cache(weights: str, frames_root: Path, imgsz: int) -> pd.DataFrame:
    """Run the detector once per test frame; cache all boxes + geometry."""
    from ultralytics import YOLO

    df = load_snv3d_csv(DATA / "annotations" / "SNv3D.csv")
    test = set(load_split(DATA / "splits" / "SNv3D-test.txt"))
    tb = df[
        df["frame_id"].isin(test) & df["ball_3D"].notna() & df["calibration"].notna()
    ].copy()

    model = YOLO(weights)
    rows = []
    for _, r in tb.iterrows():
        match, frame = r["frame_id"].rsplit("/", 1)
        img = frames_root / match / (frame + ".png")
        if not img.exists():
            continue
        res = model.predict(str(img), imgsz=imgsz, conf=0.01, verbose=False)[0]
        fh, fw = res.orig_shape
        sx, sy = float(r["img_w"]) / fw, float(r["img_h"]) / fh  # frame -> annotation

        cam = Camera.from_soccernet(r["calibration"])
        gt = np.asarray(r["ball_3D"], dtype=float)
        rng = float(np.linalg.norm(gt - cam.position))

        # GT ball box in frame pixels (for IoU); annotation box scaled up.
        gt_box_frame = None
        if r["ball_bbox"] is not None:
            bb = np.asarray(r["ball_bbox"], dtype=float)
            gt_box_frame = np.array([bb[0] / sx, bb[1] / sy, bb[2] / sx, bb[3] / sy])

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        for b, c in zip(boxes, confs):
            box_frame = xyxy_to_xywh(b)
            iou = float(iou_xywh(box_frame, gt_box_frame)) if gt_box_frame is not None else np.nan
            rows.append(
                {
                    "frame_id": r["frame_id"],
                    "conf": float(c),
                    # box in ANNOTATION space (calibration lives there)
                    "bx": box_frame[0] * sx,
                    "by": box_frame[1] * sy,
                    "bw": box_frame[2] * sx,
                    "bh": box_frame[3] * sy,
                    "iou_gt": iou,
                    "gx": gt[0], "gy": gt[1], "gz": gt[2],
                    "range_m": rng,
                    "fx": cam.K[0, 0], "fy": cam.K[1, 1],
                    "calib_idx": r.name,
                }
            )
    cache = pd.DataFrame(rows)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache.to_parquet(CACHE)
    return cache


def sweep(cache: pd.DataFrame, calib_lookup: pd.Series) -> pd.DataFrame:
    confs = [0.25, 0.40, 0.50]
    diams = [0.21, 0.22, 0.23]
    modes = ["mean", "max", "geom"]

    results = []
    for conf_t, D, mode in itertools.product(confs, diams, modes):
        kept = cache[cache["conf"] >= conf_t]
        # Top detection per frame.
        top = kept.sort_values("conf").groupby("frame_id").tail(1)
        errs_all, errs_loc, rels_all = [], [], []
        for _, d in top.iterrows():
            cam = Camera.from_soccernet(calib_lookup.loc[d["calib_idx"]])
            box = np.array([d["bx"], d["by"], d["bw"], d["bh"]])
            center = np.array([d["bx"] + d["bw"] / 2, d["by"] + d["bh"] / 2])
            diam = apparent_diameter(box, mode)
            if diam <= 0:
                continue
            est = localize_ball_monocular(cam, center, diam, real_diameter=D)
            gt = np.array([d["gx"], d["gy"], d["gz"]])
            e = float(np.linalg.norm(est - gt))
            errs_all.append(e)
            rels_all.append(e / d["range_m"])
            if d["iou_gt"] >= 0.5:
                errs_loc.append(e)

        n_frames = cache["frame_id"].nunique()
        s_all = localization_error_stats(np.array(errs_all))
        s_loc = localization_error_stats(np.array(errs_loc))
        results.append(
            {
                "conf": conf_t, "D_real": D, "diam_mode": mode,
                "n_fired": s_all.get("n", 0),
                "det_rate": s_all.get("n", 0) / n_frames,
                "MAEm_all": s_all.get("mean_m", np.nan),
                "median_all": s_all.get("median_m", np.nan),
                "MAEpct_all": 100 * float(np.mean(rels_all)) if rels_all else np.nan,
                "P2m_all": s_all.get("p2m", np.nan),
                "n_localized": s_loc.get("n", 0),
                "MAEm_loc": s_loc.get("mean_m", np.nan),
                "median_loc": s_loc.get("median_m", np.nan),
                "P2m_loc": s_loc.get("p2m", np.nan),
            }
        )
    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DATA / "weights" / "yolo-sn-ball-opt.pt"))
    ap.add_argument("--frames", default=str(DATA / "frames" / "test"))
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "eval" / "monocular3d_sweep.csv"))
    args = ap.parse_args()

    # calibration dicts are keyed by original CSV row index (calib_idx).
    df = load_snv3d_csv(DATA / "annotations" / "SNv3D.csv")
    calib_lookup = df["calibration"]

    if args.rebuild or not CACHE.exists():
        print("building detection cache (one detector pass)...")
        cache = build_cache(args.weights, Path(args.frames), args.imgsz)
    else:
        cache = pd.read_parquet(CACHE)
    print(f"cache: {len(cache)} detections over {cache['frame_id'].nunique()} frames")

    grid = sweep(cache, calib_lookup)
    grid.to_csv(args.out, index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    best = grid.iloc[(grid["MAEm_loc"] - 4.2).abs().argmin()]
    print("\n=== configs closest to paper (MAEm 4.2 m), localization-only ===")
    print(grid.sort_values((grid["MAEm_loc"] - 4.2).abs().name or "MAEm_loc")
          .assign(gap=(grid["MAEm_loc"] - 4.2).abs())
          .sort_values("gap")
          .head(5)[["conf", "D_real", "diam_mode", "det_rate",
                    "MAEm_all", "MAEm_loc", "median_loc", "P2m_loc"]]
          .to_string(index=False))
    print(f"\nbest localization-only config: conf={best['conf']} D={best['D_real']} "
          f"mode={best['diam_mode']}  MAEm_loc={best['MAEm_loc']:.3f}m "
          f"(all={best['MAEm_all']:.3f}m)  P2m_loc={best['P2m_loc']:.3f}")
    print(f"full grid -> {args.out}")


if __name__ == "__main__":
    main()
