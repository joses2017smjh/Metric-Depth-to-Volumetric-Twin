"""Tests for the physics-constrained ballistic trajectory fitter."""

import numpy as np
import pytest

from v3d.calibration import Camera
from v3d.trajectory import GRAVITY, fit_ballistic, fit_ballistic_single_view


def _cam(pos, f=2000.0, w=1280, h=720):
    cz = -np.asarray(pos, float)
    cz /= np.linalg.norm(cz)
    up = np.array([0.0, 0.0, -1.0])  # z-down world
    cx = np.cross(up, cz)
    cx /= np.linalg.norm(cx)
    cy = np.cross(cz, cx)
    R = np.stack([cx, cy, cz])
    K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])
    return Camera(K=K, R=R, position=np.asarray(pos, float))


def _fly(X0, V0, ts):
    return X0 + np.outer(ts, V0) + 0.5 * np.outer(ts**2, GRAVITY)


def test_single_view_recovers_clean_trajectory():
    cam = _cam([0, 60, -15])
    X0, V0 = np.array([5.0, -3.0, -8.0]), np.array([2.0, 4.0, -6.0])
    ts = np.arange(9) / 25.0
    pix = np.array([cam.project(X)[0] for X in _fly(X0, V0, ts)])
    tr = fit_ballistic_single_view(cam, ts, pix)
    np.testing.assert_allclose(tr.X0, X0, atol=1e-4)
    np.testing.assert_allclose(tr.V0, V0, atol=1e-3)
    assert tr.rms_reproj_px < 1e-3


def test_position_evaluates_parabola():
    cam = _cam([0, 60, -15])
    X0, V0 = np.array([1.0, 2.0, -5.0]), np.array([0.0, 0.0, -4.0])
    ts = np.arange(7) / 25.0
    pix = np.array([cam.project(X)[0] for X in _fly(X0, V0, ts)])
    tr = fit_ballistic_single_view(cam, ts, pix)
    # At apex-ish time the fitted parabola matches the true one.
    for t in ts:
        np.testing.assert_allclose(tr.position(t)[0], X0 + V0 * t + 0.5 * GRAVITY * t**2,
                                   atol=1e-3)


def test_multiview_instant_is_triangulation():
    # Same time, several cameras -> ballistic fit reduces to triangulation.
    cams = [_cam([0, 60, -15]), _cam([40, 40, -18]), _cam([-35, 45, -16])]
    X = np.array([5.0, -3.0, -6.0])
    ts = np.zeros(3)
    pix = np.array([c.project(X)[0] for c in cams])
    tr = fit_ballistic(cams, ts, pix, refine=True)
    np.testing.assert_allclose(tr.X0, X, atol=1e-3)


def test_noise_degrades_gracefully():
    cam = _cam([0, 60, -15])
    X0, V0 = np.array([5.0, -3.0, -8.0]), np.array([2.0, 4.0, -6.0])
    ts = np.arange(9) / 25.0
    Xs = _fly(X0, V0, ts)
    pix = np.array([cam.project(X)[0] for X in Xs])
    rng = np.random.default_rng(0)
    tr = fit_ballistic_single_view(cam, ts, pix + rng.normal(0, 1.0, pix.shape))
    # Single-view depth is noise-sensitive but should stay within a few meters.
    assert np.linalg.norm(tr.X0 - X0) < 8.0
    assert tr.rms_reproj_px < 3.0
