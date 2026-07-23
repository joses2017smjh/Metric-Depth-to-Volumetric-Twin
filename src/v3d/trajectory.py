"""Physics-constrained 3D ball localization from a single camera's 2D track.

The Phase-1 monocular baseline gets depth from apparent ball size, which is
fragile (diameter-dominated, ~4 m error). This module gets depth from *motion
under gravity* instead: a ball in free flight follows

    X(t) = X0 + V0 t + 1/2 a t^2 ,   a = gravity (world frame),

so a window of 2D observations from ONE fixed camera constrains the full 3D
trajectory — no size prior needed. A single view of a straight, constant-
velocity path is depth-ambiguous; the known gravitational acceleration is
exactly what breaks that ambiguity, so the method is informative precisely when
the ball is accelerating (airborne).

ISSIA / SoccerNet world frame: z points DOWN (all cameras sit at z ~= -21 m,
i.e. ~21 m above the pitch), ground at z = 0, so gravity is a = (0, 0, +g).

Linear solve
------------
X(t) is linear in the unknown u = [X0, V0] (6-vector) once the gravity term is
moved to a known offset. Cross-multiplying the projection removes the
perspective division and leaves, per observation, two equations that are LINEAR
in u:

    (x * P3 - P1) . [X(t); 1] = 0
    (y * P3 - P2) . [X(t); 1] = 0

with [X(t); 1] = M(t) u + c(t),  M(t) = [[I, tI], [0, 0]],
c(t) = [1/2 a t^2; 1]. Stacking >= 3 observations and solving least squares
gives an algebraic estimate; `refine=True` then minimizes true reprojection
error with Levenberg-Marquardt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from v3d.calibration import Camera

GRAVITY = np.array([0.0, 0.0, 9.81])  # world frame, z-down


@dataclass
class BallisticTrajectory:
    X0: np.ndarray          # position at t=0 (window reference), meters
    V0: np.ndarray          # velocity at t=0, m/s
    gravity: np.ndarray     # acceleration used, m/s^2
    rms_reproj_px: float    # RMS reprojection error over the fit window
    n_obs: int              # number of 2D observations used
    condition: float        # condition number of the linear system (ill-posed -> large)

    def position(self, t: float | np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return self.X0 + np.outer(t, self.V0) + 0.5 * np.outer(t**2, self.gravity)


def _linear_solve(cams, times, pixels, gravity):
    """Algebraic (DLT-style) least-squares solve for u = [X0, V0]."""
    rows, rhs = [], []
    for cam, t, (x, y) in zip(cams, times, pixels):
        P = cam.projection_matrix()
        P1, P2, P3 = P[0], P[1], P[2]
        # M(t): 4x6 maps u -> [X(t);1] linear part; c(t): known offset incl gravity.
        M = np.zeros((4, 6))
        M[:3, :3] = np.eye(3)
        M[:3, 3:] = t * np.eye(3)
        c = np.array([0.5 * gravity[0] * t**2, 0.5 * gravity[1] * t**2,
                      0.5 * gravity[2] * t**2, 1.0])
        for Prow, obs in ((P1, x), (P2, y)):
            a_lin = (obs * P3 - Prow)      # 1x4
            rows.append(a_lin @ M)         # 1x6
            rhs.append(-a_lin @ c)         # scalar
    A = np.asarray(rows)
    b = np.asarray(rhs)
    u, *_ = np.linalg.lstsq(A, b, rcond=None)
    cond = float(np.linalg.cond(A))
    return u[:3], u[3:], cond


def init_from_points(times, points, gravity):
    """Initialize (X0, V0) by regressing rough per-observation 3D points.

    points: (N,3) approximate 3D positions (e.g. from the per-frame size prior).
    Fits points(t) ~= X0 + V0 t + 1/2 g t^2 in the 3D domain (well-posed and
    robust), giving a physically sensible starting guess for the reprojection
    refinement — far more stable than the algebraic solve, whose error is
    depth-unstable for a single view.
    """
    times = np.asarray(times, float)
    pts = np.asarray(points, float)
    resid = pts - 0.5 * np.outer(times**2, gravity)  # remove known gravity term
    A = np.vstack([np.ones_like(times), times]).T     # (N,2): [1, t]
    coef, *_ = np.linalg.lstsq(A, resid, rcond=None)   # (2,3): rows X0, V0
    return coef[0], coef[1]


def _reproject_residuals(u, cams, times, pixels, gravity, prior=None):
    """Reprojection residuals (px), optionally with a soft prior toward `prior`.

    A single view's reprojection error is nearly flat along the depth ray, so
    without a prior the depth is unconstrained and the optimizer drifts. The
    prior residual `w * (u - prior.u0)` (units px per meter / per m/s) resolves
    that null direction using the size-prior initialization, without meaningfully
    biasing the well-constrained lateral directions.
    """
    X0, V0 = u[:3], u[3:]
    res = []
    for cam, t, (x, y) in zip(cams, times, pixels):
        X = X0 + V0 * t + 0.5 * gravity * t**2
        px = cam.project(X)[0]
        res.extend([px[0] - x, px[1] - y])
    if prior is not None:
        u0, w_pos, w_vel = prior
        res.extend(list(w_pos * (X0 - u0[:3])))
        res.extend(list(w_vel * (V0 - u0[3:])))
    return np.asarray(res)


def fit_ballistic(
    cams: list[Camera],
    times: np.ndarray,
    pixels: np.ndarray,
    gravity: np.ndarray = GRAVITY,
    refine: bool = True,
    init: np.ndarray | None = None,
    init_points: np.ndarray | None = None,
    prior_pos_px_per_m: float = 0.0,
    prior_vel_px_per_mps: float = 0.0,
) -> BallisticTrajectory:
    """Fit a ballistic 3D trajectory to 2D observations.

    cams/times/pixels are aligned lists of length N (N >= 3). For the
    single-camera case pass the same Camera repeated; for a multi-view instant
    pass different cameras at the same time. `times` are seconds relative to an
    arbitrary window reference (t=0 defines where X0 is evaluated).

    Initialization for the reprojection refinement, in priority order:
    `init` ([X0,V0] 6-vector) > `init_points` (rough per-obs 3D, e.g. size
    prior) > the algebraic linear solve.

    `prior_pos_px_per_m` / `prior_vel_px_per_mps` add a soft prior pulling the
    solution toward the init (the size prior). For a single distant view this
    is what pins the otherwise-unconstrained depth; set to 0 for the
    well-conditioned multi-view case.
    """
    times = np.asarray(times, dtype=float)
    pixels = np.asarray(pixels, dtype=float)
    assert len(cams) == len(times) == len(pixels) >= 3, "need >= 3 observations"

    _, _, cond = _linear_solve(cams, times, pixels, gravity)
    if init is not None:
        X0, V0 = np.asarray(init[:3], float), np.asarray(init[3:], float)
    elif init_points is not None:
        X0, V0 = init_from_points(times, init_points, gravity)
    else:
        X0, V0, _ = _linear_solve(cams, times, pixels, gravity)

    u0 = np.concatenate([X0, V0])
    prior = None
    if prior_pos_px_per_m > 0 or prior_vel_px_per_mps > 0:
        prior = (u0.copy(), prior_pos_px_per_m, prior_vel_px_per_mps)

    if refine:
        from scipy.optimize import least_squares

        method = "lm" if prior is None else "trf"
        sol = least_squares(
            _reproject_residuals, u0,
            args=(cams, times, pixels, gravity, prior), method=method, max_nfev=200,
        )
        X0, V0 = sol.x[:3], sol.x[3:]

    res = _reproject_residuals(np.concatenate([X0, V0]), cams, times, pixels, gravity)
    rms = float(np.sqrt(np.mean(res**2)))
    return BallisticTrajectory(X0=X0, V0=V0, gravity=gravity,
                               rms_reproj_px=rms, n_obs=len(times), condition=cond)


def fit_ballistic_single_view(
    cam: Camera,
    times: np.ndarray,
    pixels: np.ndarray,
    gravity: np.ndarray = GRAVITY,
    refine: bool = True,
    init_points: np.ndarray | None = None,
    prior_pos_px_per_m: float = 0.0,
    prior_vel_px_per_mps: float = 0.0,
) -> BallisticTrajectory:
    """Convenience wrapper: one fixed camera, a time series of ball pixels."""
    return fit_ballistic([cam] * len(times), times, pixels, gravity,
                         refine=refine, init_points=init_points,
                         prior_pos_px_per_m=prior_pos_px_per_m,
                         prior_vel_px_per_mps=prior_vel_px_per_mps)
