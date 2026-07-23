"""Track 4: validate the triangulation covariance model against Monte Carlo.

The first-order covariance Cov = sigma_px^2 (J^T J)^-1 is only useful if it is
*calibrated*: the predicted sigma must match the spread you actually get when
pixel noise is injected. These tests check that, and check that parallax
behaves as the conditioning knob the theory says it is.
"""

import numpy as np
import pytest

from v3d.calibration import Camera
from v3d.uncertainty import (
    parallax_angle,
    projection_jacobian,
    triangulate_with_covariance,
)


def _cam(pos, f=2000.0, w=1280, h=720):
    cz = -np.asarray(pos, float)
    cz /= np.linalg.norm(cz)
    up = np.array([0.0, 0.0, -1.0])
    cx = np.cross(up, cz)
    cx /= np.linalg.norm(cx)
    cy = np.cross(cz, cx)
    R = np.stack([cx, cy, cz])
    K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])
    return Camera(K=K, R=R, position=np.asarray(pos, float))


def test_jacobian_matches_finite_difference():
    cam = _cam([0, 60, -15])
    X = np.array([5.0, -3.0, -2.0])
    J = projection_jacobian(cam, X)
    eps = 1e-5
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps
        num = (cam.project(X + d)[0] - cam.project(X - d)[0]) / (2 * eps)
        np.testing.assert_allclose(J[:, k], num, rtol=1e-4, atol=1e-4)


def test_parallax_geometry():
    X = np.zeros(3)
    a, b = _cam([0, 50, -10]), _cam([50, 0, -10])
    # Rays to two perpendicular-ish camera positions.
    ang = parallax_angle(a, b, X)
    assert 60 < ang < 120
    # A camera pair almost in line with the point has tiny parallax.
    c, d = _cam([0, 50, -10]), _cam([0, 51, -10.2])
    assert parallax_angle(c, d, X) < 5


@pytest.mark.parametrize("cam_positions,label", [
    ([[0, 60, -15], [50, 30, -18]], "wide baseline"),
    ([[0, 60, -15], [6, 59, -15.5]], "narrow baseline"),
])
def test_predicted_sigma_matches_monte_carlo(cam_positions, label):
    """Predicted sigma must match the empirical spread under pixel noise."""
    cams = [_cam(p) for p in cam_positions]
    X_true = np.array([4.0, -2.0, -1.0])
    pix_true = np.array([c.project(X_true)[0] for c in cams])
    sigma_px = 1.0

    pred = triangulate_with_covariance(cams, pix_true, pixel_sigma=sigma_px)

    rng = np.random.default_rng(0)
    samples = []
    for _ in range(400):
        noisy = pix_true + rng.normal(0, sigma_px, pix_true.shape)
        r = triangulate_with_covariance(cams, noisy, pixel_sigma=sigma_px)
        samples.append(r.point)
    emp = np.asarray(samples)
    emp_sigma = float(np.sqrt(np.trace(np.cov(emp.T))))

    # Calibrated within a factor of ~1.5 (first-order model, finite samples).
    ratio = pred.sigma_m / emp_sigma
    assert 0.67 < ratio < 1.5, f"{label}: predicted {pred.sigma_m:.3f} vs empirical {emp_sigma:.3f}"


def test_narrow_baseline_has_larger_sigma_and_smaller_parallax():
    X = np.array([4.0, -2.0, -1.0])
    wide = [_cam([0, 60, -15]), _cam([50, 30, -18])]
    narrow = [_cam([0, 60, -15]), _cam([6, 59, -15.5])]
    rw = triangulate_with_covariance(wide, np.array([c.project(X)[0] for c in wide]))
    rn = triangulate_with_covariance(narrow, np.array([c.project(X)[0] for c in narrow]))
    assert rn.parallax_deg < rw.parallax_deg
    assert rn.sigma_m > rw.sigma_m
    # Both have near-zero reprojection error despite wildly different certainty:
    # this is exactly the trap the paper flags.
    assert rw.rms_reproj_px < 1e-6 and rn.rms_reproj_px < 1e-6


def test_per_view_sigma_scales_covariance():
    X = np.array([4.0, -2.0, -1.0])
    cams = [_cam([0, 60, -15]), _cam([50, 30, -18])]
    pix = np.array([c.project(X)[0] for c in cams])
    a = triangulate_with_covariance(cams, pix, pixel_sigma=1.0)
    b = triangulate_with_covariance(cams, pix, pixel_sigma=2.0)
    # Covariance scales with sigma^2 -> sigma_m doubles.
    np.testing.assert_allclose(b.sigma_m / a.sigma_m, 2.0, rtol=1e-6)
