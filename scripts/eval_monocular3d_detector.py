"""Phase 1: monocular 3D ball localization from DETECTED boxes (paper's MAEm).

The paper's single-image baseline runs the ball detector, then estimates the
3D position of the top detection from its apparent size. This script does the
same end-to-end thing so the number is comparable to the paper's MAEm (4.2 m):

    detect ball (YOLO) -> box in frame pixels
    -> scale box into annotation space (calibration lives there)
    -> range = f * D_real / apparent_diameter ; back-project center ray
    -> 3D error vs triangulated ball_3D.

Reports MAEm (mean Euclidean error, meters), median, and MAE% (error over
range-to-camera). Frames where the detector fires but far from the ball are
included — that is part of the honest end-to-end error.

Usage:
    python scripts/eval_monocular3d_detector.py --weights data/weights/yolo-sn-ball-opt.pt
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


def xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return np.array([x1, y1, x2 - x1, y2 - y1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DATA / "weights" / "yolo-sn-ball-opt.pt"))
    ap.add_argument("--frames", default=str(DATA / "frames" / "test"))
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--diam", choices=["mean", "max", "geom"], default="mean")
    ap.add_argument("--out", default=str(ROOT / "eval" / "monocular3d_from_detector.csv"))
    args = ap.parse_args()

    from ultralytics import YOLO

    df = load_snv3d_csv(DATA / "annotations" / "SNv3D.csv")
    test = set(load_split(DATA / "splits" / "SNv3D-test.txt"))
    tb = df[
        df["frame_id"].isin(test)
        & df["ball_3D"].notna()
        & df["calibration"].notna()
    ].copy()

    model = YOLO(args.weights)
    frames_root = Path(args.frames)

    records = []
    errs = []
    rel = []
    n_miss = 0
    for _, r in tb.iterrows():
        match, frame = r["frame_id"].rsplit("/", 1)
        img = frames_root / match / (frame + ".png")
        if not img.exists():
            continue
        res = model.predict(str(img), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        if len(res.boxes) == 0:
            n_miss += 1
            records.append({"frame_id": r["frame_id"], "err_m": np.nan, "detected": False})
            continue
        fh, fw = res.orig_shape
        # Highest-confidence detection.
        confs = res.boxes.conf.cpu().numpy()
        best = int(np.argmax(confs))
        box_frame = xyxy_to_xywh(res.boxes.xyxy[best].cpu().numpy())

        # Scale box from frame pixels into annotation space (where calibration lives).
        sx, sy = float(r["img_w"]) / fw, float(r["img_h"]) / fh
        box_annot = np.array(
            [box_frame[0] * sx, box_frame[1] * sy, box_frame[2] * sx, box_frame[3] * sy]
        )
        center = np.array([box_annot[0] + box_annot[2] / 2, box_annot[1] + box_annot[3] / 2])
        diam = apparent_diameter(box_annot, args.diam)

        cam = Camera.from_soccernet(r["calibration"])
        est = localize_ball_monocular(cam, center, diam)
        gt = r["ball_3D"]
        e = float(np.linalg.norm(est - gt))
        rng = float(np.linalg.norm(gt - cam.position))
        errs.append(e)
        rel.append(e / rng)
        records.append(
            {"frame_id": r["frame_id"], "err_m": e, "rel": e / rng, "detected": True}
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out, index=False)

    stats = localization_error_stats(np.array(errs))
    print(f"monocular 3D from detector ({Path(args.weights).name}) on SNv3D-test")
    print(f"frames with a ball_3D GT: {len(tb)}   detector fired: {len(errs)}   missed: {n_miss}")
    if stats.get("n"):
        print(f"MAEm (mean):   {stats['mean_m']:.3f} m   [paper baseline 4.2 m]")
        print(f"median:        {stats['median_m']:.3f} m")
        print(f"p90:           {stats['p90_m']:.3f} m")
        print(f"MAE% (mean):   {100*np.mean(rel):.2f} %   [paper 5.2-5.5%]")
    print(f"per-frame CSV -> {out}")


if __name__ == "__main__":
    main()
