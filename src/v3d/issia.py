"""Loaders for ISSIA-3D: six fixed, calibrated cameras around one pitch.

ISSIA is the "easy mode" testbed: cameras never move, so calibration is a
single dict per camera for the whole sequence (issia_calibration.json,
keys "cam1".."cam6", SoccerNet-calibration format).

THE FLIP TRAP (from the paper's README): the downloadable footage for
cameras 2 and 6 is horizontally mirrored, but all annotations and the
calibration are expressed in the ORIGINAL (unflipped) image orientation.
The fix is to horizontally flip the cam2/cam6 video frames once at load
time (`unflip_image`), after which every annotation and projection lines
up. `flip_x` exists for the opposite direction (mapping an annotation onto
an unfixed, still-mirrored frame) — use one or the other, never both.

ISSIA-3D.csv: one row per synchronized frame index, with per-camera columns
(suffix cam1..cam6):
* x_cam*/y_cam*  2D ball position, pixels, unflipped orientation
* err_cam*       reprojection error of ball_3D into that camera, pixels
* opt_d_cam*     optimized ball diameter, pixels
* opt_e_cam*     projection error after bbox optimization, meters
and shared columns: num_cameras, ball_3D (meters), cam_list (cameras used
in the triangulation).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from v3d.calibration import Camera
from v3d.parsing import literal, parse_ball_3d

# Cameras whose downloadable footage is horizontally mirrored.
ISSIA_FLIPPED_CAMERAS = frozenset({2, 6})


def load_issia_calibration(path: str | Path) -> dict[int, Camera]:
    """issia_calibration.json -> {camera_index: Camera} for cameras 1..6."""
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, str):
        raw = literal(raw)
    return {int(name.removeprefix("cam")): Camera.from_soccernet(c) for name, c in raw.items()}


def load_issia_csv(path: str | Path) -> pd.DataFrame:
    """Load ISSIA-3D.csv with ball_3D and cam_list parsed into objects."""
    df = pd.read_csv(path, index_col=0)
    df["ball_3D"] = df["ball_3D"].map(parse_ball_3d)
    df["cam_list"] = df["cam_list"].map(
        lambda s: [int(c) for c in literal(s)] if isinstance(s, str) else None
    )
    return df


ISSIA_FPS = 25.0  # continuous footage frame rate


def issia_camera_track(df: pd.DataFrame, camera_index: int) -> pd.DataFrame:
    """Per-camera 2D ball track for temporal methods (Track 2).

    Returns rows (sorted by frame) where camera `camera_index` observed the
    ball, with columns: frame, u, v (pixels, unflipped orientation),
    opt_d (optimized apparent diameter, pixels), and ball_3D (triangulated GT
    when available, else None). Ready to slice into time windows.
    """
    c = camera_index
    xc, yc, dc = f"x_cam{c}", f"y_cam{c}", f"opt_d_cam{c}"
    obs = df[df[xc].notna()].copy()
    out = pd.DataFrame(
        {
            "frame": obs["frame"].to_numpy(),
            "u": obs[xc].to_numpy(),
            "v": obs[yc].to_numpy(),
            "opt_d": obs[dc].to_numpy() if dc in obs else np.nan,
            "ball_3D": obs["ball_3D"].to_numpy(),
        }
    ).sort_values("frame").reset_index(drop=True)
    return out


def unflip_image(image: np.ndarray, camera_index: int) -> np.ndarray:
    """Restore the original orientation of a downloaded ISSIA frame.

    Flips horizontally for cameras 2 and 6, returns other cameras untouched.
    Apply this to every decoded frame before using annotations or Cameras.
    """
    if camera_index in ISSIA_FLIPPED_CAMERAS:
        return image[:, ::-1]
    return image


def flip_x(x: np.ndarray | float, image_width: int, camera_index: int):
    """Map unflipped-orientation x pixel coordinates onto a still-mirrored frame.

    Only needed if you choose NOT to unflip cam2/cam6 images:
    x' = (W - 1) - x. No-op for the other cameras.
    """
    if camera_index in ISSIA_FLIPPED_CAMERAS:
        return (image_width - 1) - x
    return x
