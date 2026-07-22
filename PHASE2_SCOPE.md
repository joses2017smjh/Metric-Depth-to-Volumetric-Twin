# Phase 2 — Improvement experiments (scope)

Phase 1 reproduced the paper's baselines exactly enough to measure against
(see [`eval/PHASE1_RESULTS.md`](eval/PHASE1_RESULTS.md)). Phase 2 begins the
improvements. This document scopes the two near-term tracks and the data
prerequisite that reorders them.

## The prerequisite that reorders the tracks

Both original Phase-2 tracks need **temporal sequences of ball positions**, but
the released `Frames-v3.zip` per game holds only **sparse keyframes** (each
action + its replays, ~122 frames/match) — *not* consecutive video. So neither
track can run on the SoccerNet-v3D frames as-is:

- **Track 1 — temporal ball detection** (stack N consecutive frames, or add a
  frame-difference channel): needs consecutive frames around each keyframe.
- **Track 2 — physics-constrained depth** (fit a gravity + drag ballistic arc
  over a window of 2D detections): needs a time series of ball positions.

**ISSIA-3D is the unlock.** It is genuinely continuous 25 fps video from six
fixed, calibrated cameras, and `ISSIA-3D.csv` already provides per-frame ball
positions indexed by a continuous `frame` column, with fixed per-camera
calibration in `issia_calibration.json`. That is a ready-made temporal sequence
— Track 2 can be built and validated end-to-end **with zero new downloads or
re-calibration**. Loaders already exist: `v3d.issia.load_issia_csv`,
`load_issia_calibration` (and the cam 2/6 flip handling).

## Track 2 first — physics-constrained ball depth on ISSIA

Motivation from Phase 1: monocular depth-from-size is **diameter-dominated** and
its error is ~4–5 m. A ballistic prior replaces the fragile per-frame size cue
with physics across a window.

Approach:
1. Build per-camera 2D ball tracks from `ISSIA-3D.csv` (`x_cam*`, `y_cam*`), and
   the triangulated `ball_3D` per frame as ground truth.
2. Baseline to beat: per-frame monocular localization (reuse
   `v3d.geometry.localize_ball_monocular`) → 3D error vs `ball_3D`.
3. Fit a ballistic model over a sliding window: constant gravity, optional
   quadratic drag, piecewise across bounces/kicks. Parameterize the 3D
   trajectory and fit it to the back-projected rays (not the size cue), so depth
   comes from geometry + time, not apparent diameter.
4. Compare windowed-physics 3D error vs the per-frame monocular baseline and vs
   multi-view triangulation (`v3d.geometry.triangulate_dlt`) as an upper bound.

Metrics: MAEm, median, P2m (already in `v3d.metrics.localization_error_stats`),
reported per window length. Config file + append-only `eval/track2_*.csv`.

Deliverable: a new `src/v3d/trajectory.py` (ballistic fit) + `scripts/eval_track2_issia.py`.

## Track 1 second — temporal ball detection (pending a data decision)

Needs dense frames. Two options, to decide before starting:

- **Download SoccerNet LR match videos** via the `SoccerNet` package
  (`downloadGames(files=["1.mkv","2.mkv"])`, same public password), extract short
  windows around each annotated keyframe, and build stacked-frame / frame-diff
  detector inputs. This is a **>10 GB download → confirm first** (a few GB per
  match; a handful of matches suffices for a proof-of-concept).
- **Defer** until Track 2's ISSIA pipeline proves out the temporal machinery,
  then reuse it on ISSIA video (where consecutive frames are free) before
  scaling to SoccerNet broadcast footage.

Recommended: prove the temporal stack on ISSIA via Track 2, then revisit the
SoccerNet video download for Track 1 as a deliberate, confirmed step.

## Rules carried forward

- Every experiment gets a config and writes metrics to `eval/` as append-only
  CSV; Phase 1 baselines are frozen.
- Confirm before any download over 10 GB.
- Explain the geometry (frames, units, the physics model) as it's implemented.
