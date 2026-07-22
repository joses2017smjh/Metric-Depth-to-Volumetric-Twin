"""Geometry validation: triangulation, back-projection, monocular localization.

Synthetic tests are exact; the real-data test checks that DLT triangulation of
the annotated 2D ball centers across an action + its calibrated replays
recovers the published ball_3D.
"""

from pathlib import Path

import numpy as np
import pytest

from v3d import Camera, load_snv3d_csv
from v3d.geometry import (
    D_REAL,
    apparent_diameter,
    backproject_ray,
    localize_ball_monocular,
    triangulate_dlt,
)
from v3d.metrics import evaluate_detections, iou_xywh

DATA = Path(__file__).resolve().parents[1] / "data"


def _synthetic_cam(pan_deg, position, f=2000.0, w=1280, h=720):
    """A camera looking roughly at the origin from `position`."""
    cz = np.array([0.0, 0.0, 0.0]) - np.asarray(position)
    cz = cz / np.linalg.norm(cz)
    up = np.array([0.0, 0.0, -1.0])  # z-down world
    cx = np.cross(up, cz)
    cx = cx / np.linalg.norm(cx)
    cy = np.cross(cz, cx)
    R = np.stack([cx, cy, cz])  # world->camera
    K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]])
    return Camera(K=K, R=R, position=np.asarray(position, dtype=float))


def test_triangulation_roundtrip_synthetic():
    X = np.array([5.0, -3.0, 0.0])
    cams = [
        _synthetic_cam(0, [0, 60, -15]),
        _synthetic_cam(30, [40, 40, -18]),
        _synthetic_cam(-30, [-35, 45, -16]),
    ]
    pixels = np.array([cam.project(X)[0] for cam in cams])
    Xhat = triangulate_dlt(cams, pixels)
    np.testing.assert_allclose(Xhat, X, atol=1e-6)


def test_backproject_ray_hits_point():
    cam = _synthetic_cam(0, [0, 60, -15])
    X = np.array([5.0, -3.0, 0.0])
    px = cam.project(X)[0]
    origin, direction = backproject_ray(cam, px)
    # The true point lies along the ray from the camera center.
    to_point = X - origin
    to_point /= np.linalg.norm(to_point)
    assert np.dot(to_point, direction) > 0.9999


def test_monocular_localizer_self_consistent():
    """If we synthesize the apparent diameter a ball at X would produce, the
    localizer must recover X."""
    cam = _synthetic_cam(0, [0, 60, -15])
    X = np.array([10.0, 5.0, 0.0])
    f = 0.5 * (cam.K[0, 0] + cam.K[1, 1])
    r = np.linalg.norm(X - cam.position)
    d_px = f * D_REAL / r
    center = cam.project(X)[0]
    Xhat = localize_ball_monocular(cam, center, d_px)
    np.testing.assert_allclose(Xhat, X, atol=1e-3)


def test_apparent_diameter_modes():
    box = np.array([100.0, 200.0, 20.0, 16.0])
    assert apparent_diameter(box, "mean") == 18.0
    assert apparent_diameter(box, "max") == 20.0
    assert abs(apparent_diameter(box, "geom") - np.sqrt(320)) < 1e-9


def test_iou_and_detection_metrics():
    a = np.array([0, 0, 10, 10])
    assert iou_xywh(a, a) == 1.0
    assert iou_xywh(a, np.array([100, 100, 10, 10])) == 0.0
    # One frame, one GT, one good + one bad prediction.
    res = evaluate_detections(
        [{"gt": a, "preds": [(a + np.array([1, 1, 0, 0]), 0.9),
                             (np.array([50, 50, 10, 10]), 0.3)]}],
        iou_thr=0.5,
    )
    assert res.n_gt == 1 and res.recall == 1.0
    assert res.ap > 0.9


@pytest.mark.skipif(
    not (DATA / "annotations" / "SNv3D.csv").exists(),
    reason="release data not downloaded",
)
def test_real_multiview_triangulation_recovers_ball3d():
    """Triangulate annotated 2D ball centers across frames sharing a ball_3D."""
    df = load_snv3d_csv(DATA / "annotations" / "SNv3D.csv")
    rows = df[df["ball_3D"].notna() & df["ball_bbox"].notna() & df["calibration"].notna()].copy()
    rows["key3d"] = rows["ball_3D"].map(lambda a: tuple(np.round(a, 4)))

    errs = []
    for _, grp in rows.groupby(["match", "key3d"]):
        if len(grp) < 2:
            continue
        cams, pix, gt = [], [], None
        for _, r in grp.iterrows():
            cams.append(Camera.from_soccernet(r["calibration"]))
            bx, by, bw, bh = r["ball_bbox"]
            pix.append([bx + bw / 2, by + bh / 2])
            gt = r["ball_3D"]
        Xhat = triangulate_dlt(cams, np.array(pix))
        errs.append(float(np.linalg.norm(Xhat - gt)))
        if len(errs) >= 50:
            break

    assert errs, "no multi-view ball groups found"
    med = float(np.median(errs))
    # Our DLT should land close to the published triangulation (annotation
    # noise + their possibly nonlinear refinement keep it from being exact).
    assert med < 0.5, f"median triangulation error {med:.3f} m too high"
