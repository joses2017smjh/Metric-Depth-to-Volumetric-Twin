"""Build a YOLO-format version of SNv3D-test and run official model.val().

Writes images/ symlinks + labels/*.txt (normalized cx,cy,w,h for the single
ball box, scaled from annotation space into the actual frame resolution) and
a data.yaml, then runs Ultralytics' mAP@0.5 evaluation — the canonical metric,
to cross-check our custom evaluate_detections().
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from v3d import Camera
from v3d.snv3d import load_snv3d_csv, load_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def opt_box(r):
    """Optimized (tight) ball box = raw annotation CENTER + optimized_d SIZE.

    Empirically the fine-tuned detector fires at the visual ball center (the
    raw annotation center), not the reprojected ball_3D center (which is
    ~rep_error px away), and predicts a tight box matching optimized_d rather
    than the loose raw diameter. This is the box YOLOopt was trained/evaluated
    against.
    """
    d = r.get("optimized_d")
    if r["ball_bbox"] is None or d is None or not np.isfinite(d) or d <= 0:
        return None
    bx, by, bw, bh = r["ball_bbox"]
    cx, cy = bx + bw / 2, by + bh / 2
    return np.array([cx - d / 2, cy - d / 2, d, d])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DATA / "weights" / "yolo-sn-ball-opt.pt"))
    ap.add_argument("--gt", choices=["raw", "optimized"], default="raw")
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--out", default=str(DATA / "yolo_test"))
    args = ap.parse_args()

    df = load_snv3d_csv(DATA / "annotations" / "SNv3D.csv")
    test = set(load_split(DATA / "splits" / "SNv3D-test.txt"))
    tb = df[df["frame_id"].isin(test) & df["is_ball"] & df["ball_bbox"].notna()].copy()

    out = Path(args.out)
    img_dir, lbl_dir = out / "images", out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for _, r in tb.iterrows():
        match, frame = r["frame_id"].rsplit("/", 1)
        src = DATA / "frames" / "test" / match / (frame + ".png")
        if not src.exists():
            continue
        stem = (match + "__" + frame).replace(" ", "_").replace("/", "_")
        # Real frame resolution.
        im = cv2.imread(str(src))
        fh, fw = im.shape[:2]
        sx, sy = fw / float(r["img_w"]), fh / float(r["img_h"])
        box = opt_box(r) if args.gt == "optimized" else r["ball_bbox"]
        if box is None:
            continue
        x, y, w, h = box
        cx = (x + w / 2) * sx / fw
        cy = (y + h / 2) * sy / fh
        nw, nh = w * sx / fw, h * sy / fh
        link = img_dir / (stem + ".png")
        if not link.exists():
            link.symlink_to(src.resolve())
        (lbl_dir / (stem + ".txt")).write_text(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        n += 1

    yaml = out / "data.yaml"
    yaml.write_text(
        f"path: {out.resolve()}\ntrain: images\nval: images\nnames:\n  0: ball\n"
    )
    print(f"built YOLO test set: {n} frames -> {out}")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    metrics = model.val(data=str(yaml), imgsz=args.imgsz, conf=0.001, iou=0.6,
                        verbose=False, plots=False, save_json=False)
    print(f"\nOfficial Ultralytics val ({Path(args.weights).name}, gt={args.gt}, imgsz={args.imgsz}):")
    print(f"  mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  precision:    {metrics.box.mp:.4f}")
    print(f"  recall:       {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
