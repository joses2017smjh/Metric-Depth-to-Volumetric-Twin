# Track 4 — Uncertainty-aware triangulation

The SoccerNet-v3D paper flags that low-parallax camera pairs give high 3D
uncertainty *despite low reprojection error*. This track turns that warning
into a measured quantity: propagate per-detection pixel noise through
triangulation to a full 3×3 position covariance, expose the parallax angle
that governs conditioning, and check whether the predicted σ actually predicts
error on real data.

Implementation: `src/v3d/uncertainty.py`
(`triangulate_with_covariance`, `parallax_angle`, `projection_jacobian`).
Model: `Cov(X) ≈ σ_px² (JᵀJ)⁻¹`, with the analytic pinhole Jacobian.
Per-view σ is supported, so a detector's per-detection confidence can feed
straight through into the reported 3D uncertainty.

## Headline: reprojection error cannot detect this failure

On the 1,747 released SoccerNet-v3D two-view ball annotations:

| | median reprojection error |
|---|---|
| groups with σ > 1 m | **0.97 px** |
| groups with σ ≤ 1 m | 2.16 px |

Spearman(σ, reprojection error) = **−0.188**. The most uncertain annotations
look *better* by reprojection than the well-conditioned ones. Reprojection
error is not merely a weak proxy for 3D reliability — on this data it is
**anti-correlated**, which is exactly the trap the paper warns about, now
quantified.

## Predicted σ ranks physically-impossible positions

Validation on real data without a better 3D reference: a ball cannot be
underground, 15 m in the air, or off the pitch. Those cases should concentrate
where σ is large — and they do, monotonically over ~200×:

| predicted σ | n | physically impossible | median parallax |
|---|---|---|---|
| < 0.1 m | 1545 | 0.3% | 29.0° |
| 0.1 – 0.5 m | 186 | 3.8% | 20.1° |
| 1 – 5 m | 5 | 40.0% | 0.5° |
| > 5 m | 7 | **71.4%** | 0.2° |

## A cheap parallax gate cleans the dataset

| gate | annotations kept | physically impossible | reduction |
|---|---|---|---|
| none | 100.0% | 1.20% | — |
| parallax ≥ 5° | 98.8% | 0.52% | **−57%** |
| parallax ≥ 10° | 97.9% | 0.47% | −61% |
| parallax ≥ 20° | 88.9% | 0.39% | −68% |

**Discarding 1.2% of annotations removes 57% of the impossible ones.** This is
the actionable output of the track: a one-line filter, or better, ship the σ
alongside each annotation so downstream consumers can weight rather than guess.

Scale of the issue in the released data: 2.1% of groups have parallax < 10°
(1.2% below 5°, minimum 0.03°); 0.9% have σ > 0.5 m and 0.4% exceed 5 m, up to
a maximum of 149 m.

## The covariance model is calibrated (Monte Carlo)

`tests/test_uncertainty.py` validates the model rather than assuming it:

* the analytic Jacobian matches finite differences (rtol 1e-4);
* predicted σ matches the empirical spread over 400 noise realizations, for
  both a wide and a narrow baseline (ratio within 0.67–1.5);
* a narrow baseline yields larger σ and smaller parallax while *both*
  geometries have ~zero reprojection error — the trap reproduced in a unit test;
* covariance scales as σ_px² as it must.

## ISSIA-3D: a negative result worth recording

ISSIA was the intended primary testbed, using leave-one-out **in time** as a
held-out error reference (fit a quadratic to a frame's neighbours *excluding*
it, predict its position, take the residual). Real ≥3-view groups do not exist
in either dataset — SoccerNet has none, ISSIA has 9 frames — so leave-one-out
across views was impossible.

The outcome: ISSIA's six-camera rig is **too well-conditioned to be
discriminative**. Two opposed camera banks (y = ±90 m) give a median parallax of
150.5° and a median σ of 0.06 m with almost no spread, so σ barely varies and
the rank correlation with the held-out error is weak (Spearman +0.147; parallax
−0.116). The effect is nonetheless visible where the geometry does degrade: the
worst σ-decile has median parallax 18° and **2.25× the error** of the rest
(0.09 m vs 0.04 m). Gating the least-certain 25% improves median error by ~9%.

Interpretation: uncertainty-aware triangulation matters for *broadcast* replay
geometry (SoccerNet), where camera pairs are frequently near-collinear, and is
largely redundant for a purpose-built fixed multi-camera rig (ISSIA). That is a
useful design conclusion for the volumetric-twin pipeline.

## Caveats

* `pixel_sigma = 1.0 px` is an assumption, not a measurement. σ scales linearly
  with it, so the absolute metre values shift if real detection noise differs;
  the *rankings*, correlations, and gate conclusions are unaffected.
* The covariance is first-order (Gauss-Newton). It is validated near the noise
  levels tested; for very low parallax the true posterior is banana-shaped and
  a Gaussian σ understates the tail.
* "Physically impossible" is a proxy for error, not a measurement of it. It
  catches gross failures only, so the 200× monotonicity understates how well σ
  tracks ordinary error.
* Only 12 SoccerNet groups have σ > 1 m, so the top two bins are small-sample.

## Reproduce

```bash
python scripts/eval_track4_uncertainty.py --dataset both --window 7 --pixel-sigma 1.0
pytest tests/test_uncertainty.py -v      # Monte-Carlo calibration checks
```

Outputs: `eval/track4_soccernet.csv` (per-group σ, parallax, reprojection,
implausibility) and `eval/track4_uncertainty.csv` (per-frame ISSIA with LOO
residual).
