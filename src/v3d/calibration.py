"""Camera model for the SoccerNet-calibration format.

Coordinate frames (the geometry you need to hold in your head)
--------------------------------------------------------------
World frame ("pitch frame"): origin at the center spot of the pitch, units in
meters. A FIFA pitch is 105 m x 68 m, so the playing surface spans
x in [-52.5, 52.5] (touchline direction, toward the goals) and
y in [-34, 34] (goal-line direction). Empirically in this dataset the z-axis
points DOWN: camera positions have z ~= -16 (16 m above the grass) and
triangulated ball positions sit at z ~= 0 on the ground. We verify this in
Phase 1 by reprojecting ball_3D and matching the published rep_error.

Camera frame: right-handed, origin at the camera's optical center. The
`rotation_matrix` R maps world directions into camera directions, and
`position_meters` is the camera center C expressed in the world frame.
A world point X projects as:

    x_cam = R @ (X - C) = R @ X + t,   with t = -R @ C
    pixel ~ K @ x_cam                  (divide by third coordinate)

Intrinsics K:

    K = [[fx, 0, cx],
         [0, fy, cy],
         [0,  0,  1]]

fx, fy are focal lengths in *pixels* (physical focal length divided by pixel
size); (cx, cy) is the principal point, typically the image center. The
dataset also carries OpenCV-style distortion coefficients (6 radial,
2 tangential, 4 thin-prism); in SoccerNet-v3D they are all zeros because
PnLCalib estimates a pure pinhole model, but we keep them so ISSIA or future
calibrations that do use distortion project correctly through cv2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# Keys required in a SoccerNet-calibration dict.
SOCCERNET_CALIB_KEYS = (
    "pan_degrees",
    "tilt_degrees",
    "roll_degrees",
    "x_focal_length",
    "y_focal_length",
    "principal_point",
    "position_meters",
    "rotation_matrix",
    "radial_distortion",
    "tangential_distortion",
    "thin_prism_distortion",
)


@dataclass
class Camera:
    """Pinhole camera with optional OpenCV distortion, world units in meters."""

    K: np.ndarray  # (3,3) intrinsics, pixel units
    R: np.ndarray  # (3,3) world->camera rotation
    position: np.ndarray  # (3,) camera center C in world frame, meters
    dist_coeffs: np.ndarray = field(
        default_factory=lambda: np.zeros(12)
    )  # OpenCV order: k1 k2 p1 p2 k3 k4 k5 k6 s1 s2 s3 s4
    pan_degrees: float | None = None
    tilt_degrees: float | None = None
    roll_degrees: float | None = None

    @classmethod
    def from_soccernet(cls, calib: dict) -> "Camera":
        """Build a Camera from a SoccerNet-calibration dict.

        Accepts the dict as found in Labels-v3D.json, the SNv3D.csv
        `calibration` column (after parsing), or issia_calibration.json.
        """
        fx = float(calib["x_focal_length"])
        fy = float(calib["y_focal_length"])
        cx, cy = (float(v) for v in calib["principal_point"])
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

        R = np.asarray(calib["rotation_matrix"], dtype=float)
        C = np.asarray(calib["position_meters"], dtype=float)

        # OpenCV coefficient order is k1 k2 p1 p2 k3 k4 k5 k6 s1 s2 s3 s4;
        # SoccerNet stores radial (k1..k6), tangential (p1 p2), thin-prism
        # (s1..s4) in separate lists.
        k = list(calib.get("radial_distortion") or [0.0] * 6)
        p = list(calib.get("tangential_distortion") or [0.0] * 2)
        s = list(calib.get("thin_prism_distortion") or [0.0] * 4)
        dist = np.array([k[0], k[1], p[0], p[1], k[2], k[3], k[4], k[5], *s])

        return cls(
            K=K,
            R=R,
            position=C,
            dist_coeffs=dist,
            pan_degrees=calib.get("pan_degrees"),
            tilt_degrees=calib.get("tilt_degrees"),
            roll_degrees=calib.get("roll_degrees"),
        )

    @property
    def t(self) -> np.ndarray:
        """Translation of the world->camera transform: t = -R @ C."""
        return -self.R @ self.position

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """Project world points (N,3) or (3,) meters -> pixel coordinates (N,2).

        Uses cv2.projectPoints so the full distortion model is honored
        (a no-op when all coefficients are zero, as in SoccerNet-v3D).
        """
        pts = np.atleast_2d(np.asarray(points_3d, dtype=float))
        rvec, _ = cv2.Rodrigues(self.R)
        pixels, _ = cv2.projectPoints(
            pts.reshape(-1, 1, 3), rvec, self.t, self.K, self.dist_coeffs
        )
        return pixels.reshape(-1, 2)

    def depth(self, points_3d: np.ndarray) -> np.ndarray:
        """Depth along the optical axis (camera-frame z) for world points, meters."""
        pts = np.atleast_2d(np.asarray(points_3d, dtype=float))
        return (self.R @ (pts - self.position).T)[2]

    def projection_matrix(self) -> np.ndarray:
        """P = K [R | t], the (3,4) matrix used for DLT triangulation later."""
        return self.K @ np.hstack([self.R, self.t.reshape(3, 1)])
