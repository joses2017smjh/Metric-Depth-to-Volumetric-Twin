"""Track 1: cache window detections once, then sweep aggregation parameters.

Running the detector over every frame of every window is the expensive part, so
it happens once and is cached; the aggregation hyperparameters (association
tolerance, minimum support, confidence floor, minimum speed) are then swept
cheaply over the cache.

Usage:
    python scripts/sweep_track1_temporal.py            # build cache if missing
    python scripts/sweep_track1_temporal.py --rebuild  # re-detect
"""

from __future__ import annotations

import argparse
import itertools
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from v3d.metrics import evaluate_detections, iou_xywh
from v3d.snv3d import load_snv3d_csv
from v3d.temporal import aggregate_window

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / "eval" / "track1_detcache.pkl"


def xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return np.array([x1, y1, x2 - x1, y2 - y1])


def optimized_box(row):
    d = row.get("optimized_d")
    if row["ball_bbox"] is None or d is None or not np.isfinite(d) or d <= 0:
        return None
    bx, by, bw, bh = row["ball_bbox"]
    cx, cy = bx + bw / 2, by + bh / 2
    return np.array([cx - d / 2, cy - d / 2, d, d])


def build_cache(weights, win_root: Path, W: int, imgsz: int, conf: float):
    from ultralytics import YOLO

    df = load_snv3d_csv(DATA / "annotations" / "SNv3D.csv")
    jobs = pd.read_csv(DATA / "frame_lists" / "test_action_frames.csv")
    lookup = df.set_index("frame_id")
    model = YOLO(weights)

    out = []
    for _, j in jobs.iterrows():
        stem = str(j["frame"]).removesuffix(".png")
        d = win_root / j["match"]
        if not (d / f"{stem}_{0:+d}.png").exists() or j["key"] not in lookup.index:
            continue
        r = lookup.loc[j["key"]]
        dets, shape = {}, None
        for k in range(-W, W + 1):
            p = d / f"{stem}_{k:+d}.png"
            if not p.exists():
                continue
            res = model.predict(str(p), imgsz=imgsz, conf=conf, verbose=False)[0]
            dets[k] = [(xyxy_to_xywh(b.xyxy[0].cpu().numpy()), float(b.conf[0]))
                       for b in res.boxes]
            if k == 0:
                shape = res.orig_shape
        if 0 not in dets or shape is None:
            continue
        fh, fw = shape
        gt_annot = optimized_box(r)
        gt = None
        if gt_annot is not None:
            sx, sy = fw / float(r["img_w"]), fh / float(r["img_h"])
            gt = np.array([gt_annot[0] * sx, gt_annot[1] * sy,
                           gt_annot[2] * sx, gt_annot[3] * sy])
        out.append({"key": j["key"], "dets": dets, "gt": gt})
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(out, f)
    return out


def score(cache, **kw):
    base, temp = [], []
    rec_hits = rec_total = 0
    for e in cache:
        own = e["dets"][0]
        agg = aggregate_window(e["dets"], **kw)
        base.append({"gt": e["gt"], "preds": own})
        temp.append({"gt": e["gt"], "preds": agg})
        if e["gt"] is not None:
            bi = max((iou_xywh(b, e["gt"]) for b, _ in own), default=0.0)
            ti = max((iou_xywh(b, e["gt"]) for b, _ in agg), default=0.0)
            if bi < 0.5:
                rec_total += 1
                rec_hits += ti >= 0.5
    b = evaluate_detections(base, iou_thr=0.5)
    t = evaluate_detections(temp, iou_thr=0.5)
    return b, t, (rec_hits / rec_total if rec_total else np.nan), rec_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DATA / "weights" / "yolo-sn-ball-opt.pt"))
    ap.add_argument("--windows", default=str(DATA / "frames" / "test_windows"))
    ap.add_argument("--half-window", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "eval" / "track1_sweep.csv"))
    args = ap.parse_args()

    if args.rebuild or not CACHE.exists():
        print("building detection cache (one detector pass over all windows)...")
        cache = build_cache(args.weights, Path(args.windows), args.half_window,
                            args.imgsz, args.conf)
    else:
        with open(CACHE, "rb") as f:
            cache = pickle.load(f)
    print(f"cache: {len(cache)} keyframes\n")

    b0, _, _, _ = score(cache, tol_px=12.0, min_support=3, min_conf=0.10)
    print(f"BASELINE (centre frame only): AP@0.5={b0.ap:.4f} recall={b0.recall:.4f} "
          f"prec={b0.precision:.4f} F1={b0.best_f1:.4f}\n")

    rows = []
    grid = itertools.product([8.0, 12.0, 20.0], [3, 4, 5], [0.005, 0.02, 0.05], [0.0])
    for tol, sup, mc, msp in grid:
      for refine in (True, False):
        b, t, rec, ntot = score(cache, tol_px=tol, min_support=sup,
                                min_conf=mc, min_speed_px=msp, refine=refine)
        rows.append({"tol_px": tol, "min_support": sup, "min_conf": mc, "refine": refine,
                     "AP": t.ap, "recall": t.recall, "prec": t.precision, "F1": t.best_f1,
                     "dAP": t.ap - b.ap, "drecall": t.recall - b.recall,
                     "recovered": rec, "n_missed": ntot})
    g = pd.DataFrame(rows).sort_values("AP", ascending=False)
    g.to_csv(args.out, index=False)
    pd.set_option("display.width", 200)
    print("top configs by AP@0.5:")
    print(g.head(10).round(4).to_string(index=False))
    print(f"\nfull grid -> {args.out}")


if __name__ == "__main__":
    main()
