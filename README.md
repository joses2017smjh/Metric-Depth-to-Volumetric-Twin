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

## Status

- **Phase 0 — DONE.** Environment built (PyTorch 2.13 + CUDA 13, GPU: Quadro
  RTX 8000, 46 GB), `v3d` loader package written and tested against the real
  release files. Geometry validated: reprojecting `ball_3D` with our `Camera`
  reproduces the paper's published `rep_error` to a **median of 0.002 px**
  across all 4,051 annotated frames.
- **Phase 1 — DONE.** Baselines reproduced on SNv3D-test (full writeup in
  [`eval/PHASE1_RESULTS.md`](eval/PHASE1_RESULTS.md)):

  | Metric | Ours | Paper |
  |---|---|---|
  | YOLOopt AP@0.5 | 0.813 | 0.81 |
  | YOLObase AP@0.5 | 0.686 | 0.65 |
  | Monocular MAEm (3D) | 3.76 m median / 4.51 m mean (localized) | 4.2 m |

  Test frames come from `Frames-v3.zip` via the `SoccerNet` package — **no NDA**;
  the package ships the shared-folder password. Custom AP metric cross-checks
  against official Ultralytics `val` to within 0.003. 12/12 tests pass.
- **Phase 2 / Track 2 — DONE** (physics-constrained depth on ISSIA-3D; scope in
  [`PHASE2_SCOPE.md`](PHASE2_SCOPE.md), results in
  [`eval/TRACK2_RESULTS.md`](eval/TRACK2_RESULTS.md)). Fitting a gravity-
  constrained trajectory over a 360 ms window of one camera's 2D ball track,
  anchored to the size prior, beats the per-frame size prior on 73% of windows:

  | Method | median | P2m (within 2 m) |
  |---|---|---|
  | baseline (size prior) | 3.26 m | 0.07 |
  | **physics (ballistic window)** | **2.82 m** | **0.37** |

  +13% median, **5.4× more estimates within 2 m**. Single-view depth-from-gravity
  is ill-posed alone (it diverges without the prior anchor) — see the results doc
  for that diagnosis, the window-length limit, and the tail caveat.
- **Phase 2 / Track 4 — DONE** (uncertainty-aware triangulation; results in
  [`eval/TRACK4_RESULTS.md`](eval/TRACK4_RESULTS.md)). Propagating pixel noise
  through triangulation to a 3×3 covariance shows reprojection error **cannot**
  detect low-parallax failures: SoccerNet groups with σ>1 m have *lower* median
  reprojection error (0.97 px) than well-conditioned ones (2.16 px), Spearman
  −0.188. Predicted σ ranks physically-impossible positions monotonically
  (0.3% → 71.4%), and a parallax ≥5° gate **discards 1.2% of annotations to
  remove 57% of the impossible ones**. Covariance calibration is Monte-Carlo
  validated in `tests/test_uncertainty.py`.
- **Phase 2 / Track 1 — BLOCKED.** Temporal ball detection needs dense video
  frames, and SoccerNet videos are gated behind the **NDA password** (unlike the
  frames, which ship a public password in the `SoccerNet` package). All known
  passwords return HTTP 401. To unblock: request the password via the NDA form
  at [soccer-net.org](https://www.soccer-net.org/) and set
  `downloader.password`. Keyframe→video mapping is already solved (`half` +
  `position` ms), and only **action** frames are usable (replay frames store the
  action's timestamp, not their own airtime).

Dataset as loaded: 400 matches, 5,872 action frames, 7,839 replay frames,
5,872 action→replay multi-view groups, ~81.6k player boxes with pose keypoints.
Split: 3,241 train / 810 test.

## Plan of record

### Phase 0 — Environment & data ✓
1. Scaffold the repo: `src/`, `configs/`, `notebooks/`, `eval/`, with a
   PyTorch + CUDA env (ultralytics, opencv, numpy, pandas).
2. Data loader for the `Labels-v3D.json` schema (SoccerNet-calibration format:
   pan/tilt/roll, focal, position_meters, rotation_matrix, distortion) and the
   ball CSV (`ball_bbox`, `ball_3D`, `rep_error`, JaC columns). Handle the
   ISSIA camera 2/6 horizontal flip.

The loaders live in [`src/v3d/`](src/v3d/): `calibration.py` (the `Camera`
model + coordinate-frame notes), `labels.py` (per-match `Labels-v3D.json`),
`snv3d.py` (the flat `SNv3D.csv` ball table + split files), `issia.py` (six
fixed cameras + the cam 2/6 flip), and `parsing.py` (the release files store
Python reprs, not clean JSON). Run `pytest tests/test_phase0.py` to reverify.

### Phase 1 — Reproduce the baseline (no improvements yet) ✓
3. Projection function: 3D pitch point → pixel; verify by reprojecting `ball_3D`
   and matching published `rep_error` within tolerance.
4. Run the pretrained YOLO ball detector on the SNv3D-test split and reproduce
   their detection metrics.
5. Reproduce the monocular 3D ball localization baseline (depth from apparent
   ball diameter + calibration) and its 3D error on the test set.

Reproduced and reported in [`eval/PHASE1_RESULTS.md`](eval/PHASE1_RESULTS.md).
Eval scripts in [`scripts/`](scripts/); metrics/geometry in `src/v3d/`.

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
