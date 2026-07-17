# Metric Depth to Volumetric Twin

An open-source recreation of FIFA-style semi-automated offside technology (SAOT),
starting from the SoccerNet-v3D paper (Gutiérrez-Pérez & Agudo, CVPR CVSports 2025,
[arXiv:2504.10106](https://arxiv.org/abs/2504.10106),
[mguti97/SoccerNet-v3D](https://github.com/mguti97/SoccerNet-v3D)).

The goal: go from broadcast footage → calibrated cameras → metric 3D ball and
player positions → a volumetric digital twin of the match.

## Why SoccerNet-v3D as the anchor

The paper publishes its metrics, train/test splits (`SNv3D-train.txt` /
`SNv3D-test.txt`), and baseline weights (`yolo-sn-ball-opt.pt`), so every
improvement attempted here can be measured against a real, frozen baseline.

## Data budget (~215 GB of a 250 GB allowance)

| Dataset | Size | Purpose |
|---|---|---|
| SoccerNet-v3 frames + labels | ~50 GB | Required base; `Labels-v3D.json` merges into this structure |
| SoccerNet calibration subset | ~20 GB | Extra training data for calibration experiments |
| SoccerNet tracking subset | ~40 GB | 25 fps clips with box+ID ground truth, needed for the player extension |
| ISSIA-Soccer + ISSIA-3D | ~15 GB | 6 fixed calibrated cameras — "easy mode" testbed (flip cameras 2 and 6) |
| ~20 full match videos @ 720p | ~30 GB | End-to-end demos on continuous footage |
| WorldPose subset | ~40 GB | 3D pose ground truth for benchmarking player lifting |
| SMPL-X + AMASS subset | ~20 GB | Body model + motion prior |

## Improvement tracks (each with a built-in test)

1. **Ball detection under motion blur** — replace single-frame YOLOv11 input with
   stacked consecutive frames or a motion-difference channel.
   *Test:* mAP/recall on their exact test split vs. published `yolo-sn-ball-opt.pt`.
2. **Physics-constrained monocular 3D ball localization** — fit a ballistic
   trajectory (gravity + drag, optionally Magnus) across a temporal window of 2D
   detections, so depth comes from physics rather than blur-inflated apparent diameter.
   *Test:* 3D error vs. triangulated `ball_3D` ground truth and their
   `rep_error`/`optimized_error` columns. Highest impact/effort ratio in the list.
3. **Temporally consistent calibration** — smooth pan/tilt/roll/focal across a shot
   (spline or Kalman), or a small bundle adjustment over the sequence.
   *Test:* their JaC@0.005/0.01/0.02 metrics — frames crossing the JaC 0.75
   inclusion threshold before vs. after.
4. **Uncertainty-aware triangulation** — propagate per-detection covariance,
   weight/reject low-parallax camera pairs, report calibrated confidence intervals.
   *Test:* does predicted σ correlate with actual 3D error across the annotation CSV?
5. **Extend the pipeline from ball to players ("SoccerNet-p3D")** — triangulate
   player keypoints from the same synced replay pairs + calibrations, then SMPL fits.
   *Test:* MPJPE and world-frame trajectory error vs. WorldPose ground truth.
   This is what turns "reproducing a paper" into "recreating FIFA's SAOT."

Progression: reproduce the baseline exactly (weeks 1–2) → tracks 1→2→3 as
independent quick wins (weeks 3–6) → track 5 as the main event, with track 4
woven in wherever triangulation happens.

## Plan of record

### Phase 0 — Environment & data
1. Scaffold the repo: `src/`, `configs/`, `notebooks/`, `eval/`, with a
   PyTorch + CUDA env (ultralytics, opencv, numpy, pandas).
2. Data loader for the `Labels-v3D.json` schema (SoccerNet-calibration format:
   pan/tilt/roll, focal, position_meters, rotation_matrix, distortion) and the
   ball CSV (`ball_bbox`, `ball_3D`, `rep_error`, JaC columns). Handle the
   ISSIA camera 2/6 horizontal flip.

### Phase 1 — Reproduce the baseline (no improvements yet)
3. Projection function: 3D pitch point → pixel; verify by reprojecting `ball_3D`
   and matching published `rep_error` within tolerance.
4. Run the pretrained YOLO ball detector on the SNv3D-test split and reproduce
   their detection metrics.
5. Reproduce the monocular 3D ball localization baseline (depth from apparent
   ball diameter + calibration) and its 3D error on the test set.
   **Stop and report reproduction numbers vs. the paper before continuing.**

### Phase 2 — First improvement experiments
6. Temporal ball detection (track 1): fine-tune on SNv3D-train, evaluate against
   the frozen baseline.
7. Physics-constrained ball depth (track 2): sliding-window projectile fits,
   evaluated against triangulated `ball_3D` ground truth.

## Rules

- Every experiment gets a config file and writes metrics to `eval/` as CSV;
  baseline results are never overwritten.
- Confirm before downloading anything over 10 GB.
- Geometry is explained (coordinate frames, units) as it's implemented — the
  point is to learn it, not just run it.

## Repository layout

```
src/        pipeline code (detection, calibration, triangulation, lifting)
configs/    one config per experiment
notebooks/  exploration and figures
eval/       metrics CSVs — append-only, baselines frozen
data/       datasets (gitignored)
```
