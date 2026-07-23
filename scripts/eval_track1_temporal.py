"""Track 1: temporal ball detection vs the frozen single-frame baseline.

Both methods are scored on the SAME frames (720p video frames upscaled to the
annotation resolution), so the comparison isolates the temporal contribution:

  baseline  frozen detector on the centre frame only            [Phase 1 method]
  temporal  frozen detector on all 2W+1 frames, RANSAC-linked   [Track 1]
            into a constant-velocity track, then read at k=0

Ground truth is the optimized box (raw annotation centre + `optimized_d`),
scaled into frame pixels — the same GT definition that reproduced the paper's
AP@0.5 = 0.81 in Phase 1.

Usage:
    python scripts/eval_track1_temporal.py --half-window 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from v3d import Camera
from v3d.metrics import evaluate_detections, iou_xywh
from v3d.snv3d import load_snv3d_csv
from v3d.temporal import aggregate_window

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return np.array([x1, y1, x2 - x1, y2 - y1])


def optimized_box(row):
    """Tight GT box: raw annotation centre + optimized_d (see Phase 1)."""
    d = row.get("optimized_d")
    if row["ball_bbox"] is None or d is None or not np.isfinite(d) or d <= 0:
        return None
    bx, by, bw, bh = row["ball_bbox"]
    cx, cy = bx + bw / 2, by + bh / 2
    return np.array([cx - d / 2, cy - d / 2, d, d])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DATA / "weights" / "yolo-sn-ball-opt.pt"))
    ap.add_argument("--windows", default=str(DATA / "frames" / "test_windows"))
    ap.add_argument("--half-window", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--tol-px", type=float, default=12.0)
    ap.add_argument("--min-support", type=int, default=3)
    ap.add_argument("--min-conf", type=float, default=0.10)
    ap.add_argument("--min-speed", type=float, default=0.0)
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--out", default=str(ROOT / "eval" / "track1_temporal.csv"))
    args = ap.parse_args()

    from ultralytics import YOLO

    df = load_snv3d_csv(DATA / "annotations" / "SNv3D.csv")
    jobs = pd.read_csv(DATA / "frame_lists" / "test_action_frames.csv")
    lookup = df.set_index("frame_id")

    model = YOLO(args.weights)
    win_root = Path(args.windows)

    base_frames, temp_frames, rows = [], [], []
    W = args.half_window
    for _, j in jobs.iterrows():
        stem = str(j["frame"]).removesuffix(".png")
        d = win_root / j["match"]
        center = d / f"{stem}_{0:+d}.png"
        if not center.exists():
            continue
        if j["key"] not in lookup.index:
            continue
        r = lookup.loc[j["key"]]

        # Detections on each frame of the window.
        dets_by_k = {}
        for k in range(-W, W + 1):
            p = d / f"{stem}_{k:+d}.png"
            if not p.exists():
                continue
            res = model.predict(str(p), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            dets_by_k[k] = [
                (xyxy_to_xywh(b.xyxy[0].cpu().numpy()), float(b.conf[0])) for b in res.boxes
            ]
            if k == 0:
                fh, fw = res.orig_shape
        if 0 not in dets_by_k:
            continue

        gt_annot = optimized_box(r)
        gt = None
        if gt_annot is not None:
            sx, sy = fw / float(r["img_w"]), fh / float(r["img_h"])
            gt = np.array([gt_annot[0] * sx, gt_annot[1] * sy, gt_annot[2] * sx, gt_annot[3] * sy])

        own = dets_by_k[0]
        agg = aggregate_window(dets_by_k, tol_px=args.tol_px, min_support=args.min_support,
                               min_conf=args.min_conf, min_speed_px=args.min_speed,
                               refine=not args.no_refine)

        base_frames.append({"gt": gt, "preds": own})
        temp_frames.append({"gt": gt, "preds": agg})

        def best_iou(preds):
            return max((iou_xywh(b, gt) for b, _ in preds), default=0.0) if gt is not None else np.nan

        rows.append({
            "key": j["key"], "n_window": len(dets_by_k),
            "n_det_center": len(own), "n_det_agg": len(agg),
            "base_best_iou": best_iou(own), "temp_best_iou": best_iou(agg),
            "recovered": len(agg) > len(own),
        })

    res = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)

    b = evaluate_detections(base_frames, iou_thr=args.iou_thr)
    t = evaluate_detections(temp_frames, iou_thr=args.iou_thr)

    print(f"Track 1 — temporal ball detection (window +/-{W} frames = "
          f"{(2*W+1)*40} ms, {len(res)} action keyframes)")
    print("Both scored on identical 720p-derived frames.\n")
    print(f"{'':10s} {'AP@0.5':>8s} {'recall':>8s} {'prec':>8s} {'F1':>8s}")
    print(f"{'baseline':10s} {b.ap:8.4f} {b.recall:8.4f} {b.precision:8.4f} {b.best_f1:8.4f}")
    print(f"{'temporal':10s} {t.ap:8.4f} {t.recall:8.4f} {t.precision:8.4f} {t.best_f1:8.4f}")
    print(f"{'delta':10s} {t.ap-b.ap:+8.4f} {t.recall-b.recall:+8.4f} "
          f"{t.precision-b.precision:+8.4f} {t.best_f1-b.best_f1:+8.4f}")

    hit_b = (res.base_best_iou >= args.iou_thr).mean()
    hit_t = (res.temp_best_iou >= args.iou_thr).mean()
    print(f"\nlocalization rate (any prediction IoU>={args.iou_thr}):")
    print(f"  baseline {hit_b:.3f}   temporal {hit_t:.3f}   ({hit_t-hit_b:+.3f})")
    missed = res[res.base_best_iou < args.iou_thr]
    if len(missed):
        rec = (missed.temp_best_iou >= args.iou_thr).mean()
        print(f"  of {len(missed)} frames the baseline missed, temporal recovered "
              f"{rec:.1%}")
    print(f"\nper-frame CSV -> {args.out}")


if __name__ == "__main__":
    main()
