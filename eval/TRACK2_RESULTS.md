# Track 2 — Physics-constrained ball depth (ISSIA-3D)

Improvement track 2: replace the fragile per-frame *size prior* for monocular
depth with *motion under gravity* across a temporal window. Evaluated on
ISSIA-3D (continuous 25 fps footage, six fixed calibrated cameras), scoring
single-camera 3D estimates against the multi-view triangulated `ball_3D`.

Method: `src/v3d/trajectory.py` fits `X(t) = X0 + V0 t + ½ a t²` (a = gravity)
to a window of one camera's 2D ball observations, minimizing reprojection error.
Baseline: the Phase-1 monocular size prior (`range = f·D/opt_d`).

## Headline result

Gated to windows that carry depth information (in-image ball displacement
≥ 25 px) and whose fit explains the observations (reprojection RMS ≤ 3 px) —
**73% of all windows**, cameras 3 and 4, window = 9 frames (360 ms):

| Method | median | MAEm | P2m (within 2 m) |
|---|---|---|---|
| baseline (size prior) | 3.26 m | 3.23 m | 0.07 |
| **physics (ballistic window)** | **2.82 m** | 13.51 m | **0.37** |

**+13% median error, and 5.4× more estimates within 2 m.** The mean is worse
because the physics estimator is bimodal (see caveats).

## What the experiment established

1. **Single-view depth-from-gravity is ill-posed on its own.** Without a prior
   the fit diverges catastrophically (median error 170 m at a 9-frame window,
   30 km at 21 frames). The reason is geometric: a single view's reprojection
   error is nearly flat along the depth ray, and at ISSIA's ~100 m camera range
   the gravity drop over 360 ms is only ~0.13 m — far below pixel noise. The
   optimizer simply walks down that null direction.
2. **Anchoring fixes it.** Initializing from the per-frame size prior and adding
   a soft prior (0.3 px per metre) resolves the null direction while leaving the
   well-constrained lateral directions free. Physics then *refines* the size
   prior rather than replacing it. Prior weight matters: 0 diverges, 0.3 is the
   sweet spot, 1.0 over-constrains back toward the baseline.
3. **Short windows win.** 9 frames (360 ms) is optimal; 15 and 21 frames are
   progressively *worse* (median 2.8 → 14.9 → 45.0 m at prior 0.3). The pure
   free-flight model breaks as soon as a window spans a bounce, a kick, or
   ground contact — real ball motion is only piecewise ballistic.
4. **Fit residual is a non-monotonic quality signal.** Accuracy peaks for
   reprojection RMS in [1, 2) px (median 1.94 m, P2m 0.51) and is *worse* below
   0.5 px (median 3.48 m). Near-zero residual means the ball barely moved, so
   the window carries no depth information and the fit returns the biased init.
   Ball motion, not residual, is the right gate.
5. **The baseline has a systematic bias.** Size-prior errors cluster tightly at
   ~3.2 m with P2m 0.07 — a near-constant offset, not noise. This matches the
   Phase-1 finding that monocular 3D error is diameter-dominated, and it is why
   physics can beat it in P2m so decisively while barely moving the median.

Error vs. in-image ball motion (why the gate exists):

| displacement | baseline median / P2m | physics median / P2m |
|---|---|---|
| < 10 px | 3.48 / 0.00 | 5.83 / 0.14 |
| 10–25 px | 3.35 / 0.00 | 2.92 / 0.26 |
| 25–50 px | 3.37 / 0.05 | **2.78 / 0.38** |
| 50–100 px | 3.14 / 0.05 | 2.96 / 0.37 |
| > 100 px | 2.98 / 0.14 | 4.37 / 0.19 |

Physics wins for moderate motion, loses when the ball is nearly static (no depth
signal) or very fast (window leaves free flight).

## Caveats — stated, not hidden

* **The mean is tail-dominated (13.51 m vs baseline 3.23 m).** The physics
  estimator is bimodal: far more very-good estimates *and* a tail of failures.
  Median and P2m are the meaningful statistics; anyone needing a bounded worst
  case should keep the size prior as a fallback when the gate rejects a window.
* **Per-camera split is uneven.** cam3: physics 3.12 m median / P2m 0.33 vs
  baseline 3.55 / 0.04 (clear win). cam4: physics 3.61 / 0.28 vs baseline
  2.86 / 0.09 (median loss, P2m win).
* **The airborne split did not behave as predicted.** Gravity was expected to
  help most when `|z| > 1 m`, but airborne and near-ground windows perform
  similarly (P2m 0.25 vs 0.34). At 100 m range the gravity signal is weak
  regardless; most of the gain comes from temporal smoothing of the noisy
  per-frame size estimates rather than from the gravity term itself.
* **Ground truth is itself triangulated** `ball_3D`, and ISSIA triangulation has
  its own low-parallax failures (some `z` values are physically impossible,
  e.g. 10 m underground). Part of the residual error is GT error — which is
  exactly what improvement track 4 (uncertainty-aware triangulation) targets.

## Reproduce

```bash
python scripts/eval_track2_issia.py --window 9 --prior-pos 0.3 --cameras 3 4
# sweep window length / prior weight to reproduce findings 2 and 3
python scripts/eval_track2_issia.py --window 21 --prior-pos 0.0   # diverges
```

Per-window results: `eval/track2_issia.csv` (columns include `err_phys`,
`err_base`, `rms_reproj`, `pix_disp`, `speed_mps`, `z_gt`, `camera`).
