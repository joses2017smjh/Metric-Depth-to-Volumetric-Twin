"""Tests for temporal aggregation of ball detections (Track 1)."""

import numpy as np

from v3d.temporal import aggregate_window, box_center, fit_track


def _box(cx, cy, s=14.0):
    return np.array([cx - s / 2, cy - s / 2, s, s])


def _linear_window(p0=(500.0, 300.0), v=(20.0, -8.0), W=4, conf=0.8, skip=()):
    """Detections of a constant-velocity ball, optionally skipping frames."""
    out = {}
    for k in range(-W, W + 1):
        if k in skip:
            out[k] = []
            continue
        c = np.array(p0) + np.array(v) * k
        out[k] = [(_box(*c), conf)]
    return out


def test_fit_track_recovers_linear_motion():
    dets = _linear_window()
    tr = fit_track(dets)
    assert tr is not None
    np.testing.assert_allclose(tr.p0, [500.0, 300.0], atol=1e-6)
    np.testing.assert_allclose(tr.velocity, [20.0, -8.0], atol=1e-6)
    assert tr.support == 9


def test_track_ignores_outlier_detections():
    dets = _linear_window()
    # Add distractors (e.g. a player's white sock) far from the real track.
    for k in dets:
        dets[k].append((_box(50.0 + 3 * k, 700.0), 0.6))
    tr = fit_track(dets)
    assert tr is not None
    np.testing.assert_allclose(tr.p0, [500.0, 300.0], atol=1.0)


def test_recovers_ball_missed_on_centre_frame():
    """The case that matters: detector fires on neighbours but not on k=0."""
    dets = _linear_window(skip=(0,))
    assert dets[0] == []
    out = aggregate_window(dets)
    assert len(out) == 1, "should emit an interpolated detection"
    box, conf = out[0]
    np.testing.assert_allclose(box_center(box), [500.0, 300.0], atol=1e-6)
    assert 0 < conf <= 1.0


def test_centre_detection_is_boosted_when_track_agrees():
    dets = _linear_window(conf=0.4)
    out = aggregate_window(dets)
    confs = [c for _, c in out]
    assert max(confs) > 0.4, "temporal agreement should raise confidence"


def test_no_track_leaves_detections_untouched():
    # Random, non-linear detections give no consistent constant-velocity track.
    rng = np.random.default_rng(0)
    dets = {k: [(_box(*rng.uniform(0, 1000, 2)), 0.5)] for k in range(-4, 5)}
    out = aggregate_window(dets, tol_px=3.0, min_support=4)
    assert out == dets[0]


def test_empty_window_is_safe():
    assert aggregate_window({0: []}) == []
    assert fit_track({0: [], 1: []}) is None
