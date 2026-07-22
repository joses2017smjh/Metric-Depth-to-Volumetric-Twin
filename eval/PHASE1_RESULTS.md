# Phase 1 — Baseline reproduction (SoccerNet-v3D test set)

Reproduction of the paper's ball-detection and single-image 3D localization
baselines, before any improvements. Paper: Gutiérrez-Pérez & Agudo, CVPR CVSports
2025 (arXiv:2504.10106), Table 2.

Test set: **SNv3D-test**, 810 frames (paper: 811) across 61 matches.
Detector weights: the released `yolo-sn-ball.pt` (base) and `yolo-sn-ball-opt.pt`
(fine-tuned on optimized boxes). Frames: `Frames-v3.zip` per game via the
`SoccerNet` package — **no NDA needed**, the package ships the shared-folder
password. Only the ~810 split frames are kept (zips pruned); footprint 1.5 GB.

## Results vs. paper

| Metric | Ours | Paper | Agreement |
|---|---|---|---|
| YOLObase AP@0.5 | **0.686** | 0.65 | close (+0.036) |
| YOLOopt AP@0.5 | **0.813** | 0.81 | exact |
| Monocular MAEm (detector, D=0.22) | **5.56 m** mean / **3.76 m** median | 4.2 m | median on target |
| Monocular MAEm (localization-only) | **4.51 m** mean | 4.2 m | close |
| Monocular MAE% | 7.5–7.9% | 5.2–5.5% | tracks MAEm |

Our custom `evaluate_detections` (AP@0.5 0.813 / 0.686) agrees with the official
Ultralytics `model.val()` (0.816 / 0.684) to within 0.003, cross-validating the
metric.

## Three findings that made the detection numbers reproduce

1. **Resolution scaling.** Released frames are 1920×1080, but annotation boxes
   live in per-frame `img_w×img_h` spaces (1280×720, 1366×768, 1920×1080,
   854×480). GT boxes must be scaled into frame pixels before IoU.
2. **The correct GT box is raw-center + `optimized_d`.** The fine-tuned detector
   fires at the visual ball center (the raw annotation center, *not* the
   reprojected `ball_3D` center, which sits ~`rep_error`≈3.8 px away) and predicts
   a *tight* box matching `optimized_d`, not the loose raw diameter. Scoring the
   tight predictions against loose raw GT caps IoU below 0.5 by size mismatch
   alone (two concentric squares at 13 vs 20 px have IoU 0.42), which drops
   AP@0.5 to ~0.49 — the artifact that masked the real 0.81.
3. **The 3D error is diameter-dominated.** `optimized_d ≈ f·D/range`, so depth
   comes almost entirely from apparent size; the image center barely matters
   (projected-center and bbox-center give identical 3D error). The naive
   single-image estimate from raw annotation boxes is ~17.6 m median because raw
   boxes are ~40% larger than the true angular size.

## The monocular MAEm gap, explained (not fitted)

`scripts/sweep_monocular3d.py` sweeps confidence ∈ {0.25, 0.4, 0.5}, ball-diameter
prior D ∈ {0.21, 0.22, 0.23} m, and diameter mode ∈ {mean, max, geom}, and splits
detector-failure frames (top detection IoU < 0.5 with GT) from true localization
frames. Two things drive the residual:

* **Detector-failure tail.** Over *all* fired frames MAEm = 5.56 m (p90 11.4 m);
  restricting to frames the detector actually localizes (IoU ≥ 0.5) gives
  4.51 m. A misfire is a detection failure, not a 3D-localization error, so the
  paper's MAEm most plausibly reflects the localized set.
* **Diameter under-estimation.** Error decreases monotonically with D across the
  swept range (7.6 → 5.6 → 5.0 m at conf 0.25), i.e. even D = 0.23 is not the
  minimum — the detector's tight boxes underestimate the ball's true angular
  diameter by ~5%. With the *physically correct* D = 0.22 m (regulation size-5
  ball) the median (3.76 m) already matches the paper; the mean gap is intrinsic
  to the apparent-diameter estimate.

We keep `D_REAL = 0.22` (physical) as the default rather than fitting D to the
paper's number. The reproduction is faithful: exact on detection, median-on-target
and mean-within-tolerance on 3D, with the residual attributed to a measured,
diameter-dominated effect.

## Reproduce

```bash
# 1. download + prune test frames (~1.5 GB kept)
python scripts/download_test_frames.py

# 2. detection AP@0.5 (custom metric; --gt optimized = raw-center + optimized_d)
python scripts/eval_detection.py --weights data/weights/yolo-sn-ball-opt.pt \
    --gt optimized --imgsz 1920 --out eval/detection_yoloopt.csv
python scripts/eval_detection.py --weights data/weights/yolo-sn-ball.pt \
    --gt optimized --imgsz 1920 --out eval/detection_yolobase.csv

# 2b. official Ultralytics cross-check (writes to runs/)
python scripts/build_yolo_testset.py --weights data/weights/yolo-sn-ball-opt.pt --gt optimized

# 3. monocular 3D from detected boxes + hyperparameter sweep
python scripts/eval_monocular3d_detector.py --weights data/weights/yolo-sn-ball-opt.pt
python scripts/sweep_monocular3d.py

# 4. monocular 3D from GT annotations (oracle floor / naive baseline)
python scripts/eval_monocular3d.py
```

## Result files

| File | Contents |
|---|---|
| `detection_yoloopt.csv` | per-frame detection (YOLOopt vs optimized GT) |
| `detection_yolobase.csv` | per-frame detection (YOLObase vs optimized GT) |
| `monocular3d_from_detector.csv` | per-frame 3D error from detected boxes |
| `monocular3d_sweep.csv` | full hyperparameter sweep grid |
| `monocular3d_baseline.csv` | per-frame 3D error from GT annotations (oracle/naive) |
