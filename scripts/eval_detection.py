"""Phase 1: reproduce the ball-detector metrics on SNv3D-test.

Runs a pretrained YOLOv11 ball detector over the test frames, matches
predictions to the annotated ball box per frame, and reports AP@0.5,
best-F1 recall/precision, and the operating confidence.

Usage:
    python scripts/eval_detection.py \
        --weights data/weights/yolo-sn-ball-opt.pt \
        --out eval/detection_yolo-sn-ball-opt.csv

Writes a per-frame CSV (frame_id, has_gt, n_pred, best_iou, top_conf) and
prints the summary. Baseline result files in eval/ are never overwritten
unless --out is given explicitly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from v3d import Camera
from v3d.metrics import evaluate_detections, iou_xywh
from v3d.snv3d import load_snv3d_csv, load_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return np.array([x1, y1, x2 - x1, y2 - y1])


def scale_box(box_xywh, sx, sy):
    """Scale an [x, y, w, h] box from annotation space into frame space."""
    x, y, w, h = box_xywh
    return np.array([x * sx, y * sy, w * sx, h * sy])


def optimized_box(row):
    """Optimized (tight) ball box in annotation space = raw CENTER + optimized_d SIZE.

    The fine-tuned detector fires at the visual ball center (the raw
    annotation center, not the reprojected ball_3D center) and predicts a
    tight box matching optimized_d. Returns [x, y, w, h] or None.
    """
    d = row.get("optimized_d")
    if row["ball_bbox"] is None or d is None or not np.isfinite(d) or d <= 0:
        return None
    bx, by, bw, bh = row["ball_bbox"]
    cx, cy = bx + bw / 2, by + bh / 2
    return np.array([cx - d / 2, cy - d / 2, d, d])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DATA / "weights" / "yolo-sn-ball-opt.pt"))
    ap.add_argument("--csv", default=str(DATA / "annotations" / "SNv3D.csv"))
    ap.add_argument("--split", default=str(DATA / "splits" / "SNv3D-test.txt"))
    ap.add_argument("--frames", default=str(DATA / "frames" / "test"))
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--conf", type=float, default=0.001, help="low conf floor for AP")
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--gt", choices=["raw", "optimized"], default="optimized",
                    help="ground-truth box source: raw SoccerNet-v3 box or the "
                         "reconstructed optimized (tight) box the detector was tuned for")
    ap.add_argument("--out", default=str(ROOT / "eval" / "detection_yolo-sn-ball-opt.csv"))
    args = ap.parse_args()

    from ultralytics import YOLO

    df = load_snv3d_csv(args.csv)
    test_keys = set(load_split(args.split))
    df = df[df["frame_id"].isin(test_keys)].copy()

    frames_root = Path(args.frames)
    model = YOLO(args.weights)

    per_frame = []
    rows = []
    skipped = 0
    for _, r in df.iterrows():
        match, frame = r["frame_id"].rsplit("/", 1)
        img = frames_root / match / (frame + ".png")
        if not img.exists():
            skipped += 1
            continue

        res = model.predict(str(img), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        fh, fw = res.orig_shape  # actual frame resolution the detector saw
        preds = []
        for box in res.boxes:
            xywh = xyxy_to_xywh(box.xyxy[0].cpu().numpy())
            preds.append((xywh, float(box.conf[0])))

        # Ground-truth box is in annotation space (img_w x img_h); scale it into
        # the frame's pixel space so IoU against detector boxes is meaningful.
        gt_annot = None
        if args.gt == "optimized":
            gt_annot = optimized_box(r)
        elif r["is_ball"] and r["ball_bbox"] is not None:
            gt_annot = r["ball_bbox"]

        gt = None
        if gt_annot is not None:
            sx, sy = fw / float(r["img_w"]), fh / float(r["img_h"])
            gt = scale_box(gt_annot, sx, sy)
        per_frame.append({"gt": gt, "preds": preds})

        best_iou = 0.0
        if gt is not None and preds:
            best_iou = max(iou_xywh(p[0], gt) for p in preds)
        rows.append(
            {
                "frame_id": r["frame_id"],
                "has_gt": gt is not None,
                "n_pred": len(preds),
                "best_iou": round(best_iou, 4),
                "top_conf": round(max((p[1] for p in preds), default=0.0), 4),
            }
        )

    result = evaluate_detections(per_frame, iou_thr=args.iou_thr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    print(f"weights: {Path(args.weights).name}")
    print(f"frames evaluated: {len(per_frame)}  (skipped, no image: {skipped})")
    print(f"AP@{args.iou_thr:.2f}:      {result.ap:.4f}")
    print(f"recall (bestF1):  {result.recall:.4f}")
    print(f"precision:        {result.precision:.4f}")
    print(f"best F1:          {result.best_f1:.4f}  @ conf {result.best_conf:.3f}")
    print(f"n_gt={result.n_gt}  n_pred={result.n_pred}")
    print(f"per-frame CSV -> {out}")


if __name__ == "__main__":
    main()
