"""Geometry for monocular ball localization and multi-view triangulation.

Two directions of the same camera model in v3d.calibration.Camera:

* Forward (project): world 3D -> pixel. Validated in Phase 0.
* Backward (this module): pixel + a depth/size cue -> world 3D.

Monocular ball-size-prior localization (the paper's single-image baseline)
--------------------------------------------------------------------------
A regulation size-5 soccer ball has a real diameter D_REAL = 0.22 m. Under a
pinhole camera with focal length f (pixels), a sphere at range r projects to
an apparent diameter d (pixels) of approximately

    d ~= f * D_REAL / r      =>      r ~= f * D_REAL / d

so the apparent size in one image fixes the distance to the ball. Combine
that range with the viewing ray through the ball's image center and you get a
full 3D position from a single view — no triangulation needed. This is
fragile precisely because motion blur inflates d and shrinks the estimated
range, which is what improvement track 2 (physics-constrained depth) targets.

Ray back-projection
-------------------
For pixel (u, v), the normalized camera-frame direction is
K^-1 [u, v, 1]. The camera model is x_cam = R (X - C), so a camera-frame
direction maps to world as R^T x_cam, and a point at camera-frame coords
p_cam sits at X = C + R^T p_cam.
"""

from __future__ import annotations

import numpy as np

from v3d.calibration import Camera

D_REAL = 0.22  # regulation size-5 soccer ball diameter, meters


def backproject_ray(cam: Camera, pixel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (origin, unit_direction) of the world-frame viewing ray for a pixel.

    origin is the camera center C; direction is a unit vector pointing away
    from the camera into the scene. Distortion is assumed negligible
    (SoccerNet-v3D uses a pure pinhole model); pass undistorted pixels if not.
    """
    u, v = float(pixel[0]), float(pixel[1])
    d_cam = np.linalg.inv(cam.K) @ np.array([u, v, 1.0])
    d_cam /= np.linalg.norm(d_cam)
    d_world = cam.R.T @ d_cam
    d_world /= np.linalg.norm(d_world)
    return cam.position.copy(), d_world


def apparent_diameter(bbox_xywh: np.ndarray, mode: str = "mean") -> float:
    """Apparent ball diameter in pixels from a [x, y, w, h] bbox.

    mode: "mean" (average of w, h — robust to slight non-squareness),
    "max", or "geom" (sqrt(w*h)).
    """
    w, h = float(bbox_xywh[2]), float(bbox_xywh[3])
    if mode == "max":
        return max(w, h)
    if mode == "geom":
        return float(np.sqrt(w * h))
    return 0.5 * (w + h)


def localize_ball_monocular(
    cam: Camera,
    center_px: np.ndarray,
    diameter_px: float,
    real_diameter: float = D_REAL,
) -> np.ndarray:
    """Estimate the 3D ball position (meters, world frame) from one view.

    Range from the size prior, direction from the image-center ray:
        r = f * D_real / d,   X = C + r * ray_unit.
    Uses the mean of fx, fy as f.
    """
    f = 0.5 * (cam.K[0, 0] + cam.K[1, 1])
    r = f * real_diameter / float(diameter_px)
    origin, direction = backproject_ray(cam, center_px)
    return origin + r * direction


def triangulate_dlt(cams: list[Camera], pixels: np.ndarray) -> np.ndarray:
    """Linear (DLT) triangulation of one 3D point from >= 2 views.

    pixels: (N, 2) image points, one per camera in `cams`. Solves the
    homogeneous system stacking two rows per view from x (P3^T) - (P1^T) = 0
    style constraints, via SVD. Returns the (3,) world point in meters.
    """
    assert len(cams) == len(pixels) >= 2, "need >= 2 matched views"
    rows = []
    for cam, (u, v) in zip(cams, pixels):
        P = cam.projection_matrix()
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    A = np.stack(rows)
    _, _, vt = np.linalg.svd(A)
    X = vt[-1]
    return X[:3] / X[3]


def reprojection_error(cam: Camera, point_3d: np.ndarray, pixel: np.ndarray) -> float:
    """Euclidean distance (pixels) between a projected 3D point and an observed pixel."""
    proj = cam.project(point_3d)[0]
    return float(np.linalg.norm(proj - np.asarray(pixel, dtype=float)))
