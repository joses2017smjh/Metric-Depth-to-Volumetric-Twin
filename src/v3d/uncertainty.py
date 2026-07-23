"""Uncertainty-aware triangulation (improvement track 4).

The SoccerNet-v3D paper flags that low-parallax camera pairs yield high 3D
uncertainty even when the reprojection error looks fine — a point estimate
alone hides this. This module propagates per-detection pixel noise through
triangulation to a full 3x3 position covariance, and exposes the parallax
angle that governs how well-conditioned a pair is.

Model
-----
For a 3D point X observed at pixels x_i by cameras i with projections
proj_i(.), and independent isotropic Gaussian pixel noise of std sigma_px,
the first-order (Gauss-Newton) covariance of the ML estimate is

    Cov(X) ~= sigma_px^2 (J^T J)^-1 ,

where J stacks the (2x3) Jacobians d proj_i / dX. For a pinhole camera with
M = K R and p = M (X - C):

    u = p0/p2 ,  v = p1/p2
    du/dX = (M[0] p2 - p0 M[2]) / p2^2
    dv/dX = (M[1] p2 - p1 M[2]) / p2^2

Parallax is the angle at X between the rays back to two camera centres. As it
goes to zero the two rays become parallel, J^T J becomes near-singular along
the depth direction, and the covariance blows up — which is exactly the
failure mode the paper warns about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from v3d.calibration import Camera
from v3d.geometry import triangulate_dlt


def projection_jacobian(cam: Camera, X: np.ndarray) -> np.ndarray:
    """(2,3) Jacobian d(pixel)/d(world point) for a pinhole camera at X."""
    M = cam.K @ cam.R
    p = M @ (np.asarray(X, float) - cam.position)
    p0, p1, p2 = p
    if abs(p2) < 1e-9:
        return np.zeros((2, 3))
    du = (M[0] * p2 - p0 * M[2]) / p2**2
    dv = (M[1] * p2 - p1 * M[2]) / p2**2
    return np.vstack([du, dv])


def parallax_angle(cam_a: Camera, cam_b: Camera, X: np.ndarray) -> float:
    """Angle (degrees) at X between the rays to two camera centres."""
    X = np.asarray(X, float)
    a = cam_a.position - X
    b = cam_b.position - X
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    c = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def min_parallax(cams: list[Camera], X: np.ndarray) -> float:
    """Smallest pairwise parallax angle (degrees) over the observing cameras."""
    if len(cams) < 2:
        return 0.0
    return min(
        parallax_angle(cams[i], cams[j], X)
        for i in range(len(cams))
        for j in range(i + 1, len(cams))
    )


def max_parallax(cams: list[Camera], X: np.ndarray) -> float:
    """Largest pairwise parallax angle (degrees) — the best-conditioned pair."""
    if len(cams) < 2:
        return 0.0
    return max(
        parallax_angle(cams[i], cams[j], X)
        for i in range(len(cams))
        for j in range(i + 1, len(cams))
    )


@dataclass
class TriangulationResult:
    point: np.ndarray        # (3,) triangulated position, meters
    cov: np.ndarray          # (3,3) position covariance, m^2
    sigma_m: float           # RMS position uncertainty = sqrt(trace(cov))
    sigma_major_m: float     # worst-direction uncertainty (sqrt max eigenvalue)
    parallax_deg: float      # max pairwise parallax angle
    rms_reproj_px: float
    n_views: int

    @property
    def major_axis(self) -> np.ndarray:
        """Unit vector of the worst-constrained direction (usually depth)."""
        w, v = np.linalg.eigh(self.cov)
        return v[:, int(np.argmax(w))]


def triangulate_with_covariance(
    cams: list[Camera],
    pixels: np.ndarray,
    pixel_sigma: float | np.ndarray = 1.0,
) -> TriangulationResult:
    """Triangulate and propagate pixel noise to a 3D position covariance.

    pixel_sigma may be a scalar (same noise for every view) or a per-view
    array, letting a detector's per-detection confidence feed straight through
    into the reported 3D uncertainty.
    """
    pixels = np.atleast_2d(np.asarray(pixels, float))
    n = len(cams)
    assert n == len(pixels) >= 2, "need >= 2 views"
    sig = np.full(n, float(pixel_sigma)) if np.isscalar(pixel_sigma) else np.asarray(pixel_sigma, float)

    X = triangulate_dlt(cams, pixels)

    # Weighted normal matrix: sum_i J_i^T J_i / sigma_i^2  -> Cov = inv(.)
    JtJ = np.zeros((3, 3))
    res = []
    for cam, x, s in zip(cams, pixels, sig):
        J = projection_jacobian(cam, X)
        JtJ += (J.T @ J) / max(s, 1e-9) ** 2
        res.extend(cam.project(X)[0] - x)
    try:
        cov = np.linalg.inv(JtJ)
    except np.linalg.LinAlgError:
        cov = np.full((3, 3), np.inf)

    evals = np.linalg.eigvalsh(cov)
    evals = np.clip(evals, 0.0, None)
    return TriangulationResult(
        point=X,
        cov=cov,
        sigma_m=float(np.sqrt(np.trace(cov))),
        sigma_major_m=float(np.sqrt(evals.max())),
        parallax_deg=max_parallax(cams, X),
        rms_reproj_px=float(np.sqrt(np.mean(np.asarray(res) ** 2))),
        n_views=n,
    )
