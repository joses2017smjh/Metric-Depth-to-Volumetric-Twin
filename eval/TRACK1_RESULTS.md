# Track 1 — Temporal ball detection

Improvement track 1: give the ball detector temporal context, on the premise
that a single-frame detector misses motion-blurred balls that neighbouring
frames reveal. Implemented as **test-time temporal aggregation** with the
frozen released detector (no retraining): run `yolo-sn-ball-opt.pt` on a window
of consecutive video frames, RANSAC a constant-velocity image track through the
detections, and read the ball off that track at the annotated keyframe.

Implementation: `src/v3d/temporal.py`, `scripts/extract_temporal_windows.py`,
`scripts/eval_track1_temporal.py`, `scripts/sweep_track1_temporal.py`.

## Getting the data

SoccerNet videos are NDA-gated (the frames are not). With the NDA password:

* `half` selects the video and `position` gives milliseconds into it —
  **validated against the released keyframes** at NCC ≈ 0.995 with a median
  offset of 0 ms (a few frames land at −40 ms, i.e. one frame of rounding).
* Only **action** frames are usable: replay frames store the *action's*
  timestamp, not the replay's own airtime.
* 720p halves are ~1 GB and yield only ~3 annotated frames each, so the
  pipeline streams — download, extract the ±4-frame windows, delete the video.
  Peak disk stayed ~1 GB while processing 40 videos.

Dataset: **206 action keyframes across 31 matches**, 1,854 extracted frames
(3.9 GB), the densest 40 of the 116 (match, half) files that cover the test
split's 359 action frames.

## Result: a small recall gain, no AP gain

Both methods scored on identical frames, so the comparison isolates the
temporal contribution.

| method | AP@0.5 | recall | precision | F1 |
|---|---|---|---|---|
| baseline (centre frame only) | **0.2180** | 0.2718 | 0.4870 | 0.3489 |
| temporal (best by AP) | 0.2039 | 0.2913 | 0.4580 | **0.3561** |
| temporal (best by recall) | 0.2018 | **0.3301** | 0.3696 | 0.3487 |

Temporal aggregation buys **+2 to +6 points of recall** and a marginal F1 gain,
but **AP@0.5 is consistently ~1.4 points worse** across all 54 swept
configurations. Recovered detections are frequently wrong, and each wrong one
is a false positive at moderate confidence. This is a negative result for the
method as implemented.

## Why: the bottleneck is track selection, not missing evidence

Diagnostics on the 109 keyframes (52.9%) the baseline fails:

| question | answer |
|---|---|
| centre frame has *a* detection within 40 px of GT | 85.3% |
| … within 20 px | 58.7% |
| median best centre-frame IoU on those frames | 0.072 |
| **oracle: some valid track predicts k=0 at IoU ≥ 0.5** | **78.0%** (median oracle IoU 0.725) |
| achieved by our track selection | 6.4% |

The detector nearly always proposes *something* near the ball, but those
proposals are badly localized (median IoU 0.072 — barely overlapping a ~15 px
object). Pooling them across the window into a track *does* produce a good box
78% of the time. Our scoring finds it 6% of the time. **The temporal
information is present in abundance; the selection heuristic cannot extract
it.**

Two concrete lessons from building it:

1. **Static clutter beats the ball under count-based RANSAC.** At a low
   confidence floor the detector emits ~10 boxes per frame, and a static
   distractor occupies the same pixel every frame — a perfect zero-velocity
   "track" with maximal inlier count. The first implementation therefore locked
   onto background clutter and *halved* AP (0.34 → 0.15). Ranking models by
   summed inlier **confidence**, plus size stability and fit residual, fixed the
   collapse.
2. **The track's box can be better than any single frame's.** Replacing the
   centre detection with the track estimate (`refine=True`) won on the
   103-keyframe subset but not on the full 206 — a reminder that differences of
   this size do not survive a sample-size increase.

## The premise deserves revision

Track 1 assumed single-frame detectors *miss* blurred balls. On this data the
failure is mostly **localization, not detection**: the ball is found but the box
misses the IoU@0.5 bar on a ~15 px object, exactly the size-sensitivity that
dominated Phase 1 (two concentric squares at 13 vs 20 px already score IoU
0.42). Temporal context is better spent *refining boxes* than *proposing
candidates*.

## Resolution caveat — important

The baseline AP here (0.218) is far below the Phase 1 figure (0.813) because
these frames are **720p video upscaled to 1080p**, not the native 1080p
`Frames-v3` keyframes. Same detector, same GT definition, same inference size —
only the source resolution differs. Small-object detection is extremely
resolution-sensitive, so absolute numbers here are not comparable to Phase 1;
only the baseline-vs-temporal comparison (which shares frames) is meaningful.
Any future version of this experiment should source higher-resolution video.

## Other caveats

* 206 keyframes is a modest sample; differences under ~2 points are not
  reliable, as the refine flip-flop demonstrated.
* Evaluated on 31 of 61 test matches (the densest video files), so this is a
  subset of SNv3D-test actions, not the full split.
* Constant velocity over ±4 frames is a good model for a ball in flight but
  breaks across a kick or bounce inside the window.

## Reproduce

```bash
export SOCCERNET_PASSWORD=...          # NDA-gated; never commit it
python scripts/extract_temporal_windows.py --max-files 40 --half-window 4
python scripts/sweep_track1_temporal.py --rebuild        # 54-config grid
python scripts/eval_track1_temporal.py --tol-px 8 --min-support 3 \
       --min-conf 0.05 --no-refine
```

Outputs: `eval/track1_sweep.csv` (grid), `eval/track1_temporal.csv` (per-frame).

## Where to take it next

The 78% oracle ceiling is the headline opportunity. Ranked by expected value:

1. **Better track scoring** — appearance consistency across the window, or a
   small learned scorer over candidate tracks. Closing even half the gap to the
   oracle would move AP well above baseline.
2. **The originally-planned trained variant** — a motion-difference channel or
   stacked-frame input, fine-tuned on train-split action windows. Now
   unblocked, though the data economics (~3 annotated frames per GB) make
   assembling a training set expensive.
3. **Higher-resolution source video** to lift the 0.218 baseline ceiling.
