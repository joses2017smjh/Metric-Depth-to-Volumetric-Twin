"""v3d — data loading and geometry for the SoccerNet-v3D / ISSIA-3D datasets.

Modules:
    calibration  Camera model in the SoccerNet-calibration format (K, R, t, projection)
    labels       Labels-v3D.json loader (per-match action/replay frames)
    snv3d        SNv3D.csv loader (ball 2D/3D annotations + calibration per frame)
    issia        ISSIA-3D loaders (6 fixed cameras, cameras 2/6 flip handling)
"""

from v3d.calibration import Camera
from v3d.labels import MatchLabels, load_labels_v3d
from v3d.snv3d import load_snv3d_csv, load_split
from v3d.issia import load_issia_calibration, load_issia_csv, ISSIA_FLIPPED_CAMERAS

__all__ = [
    "Camera",
    "MatchLabels",
    "load_labels_v3d",
    "load_snv3d_csv",
    "load_split",
    "load_issia_calibration",
    "load_issia_csv",
    "ISSIA_FLIPPED_CAMERAS",
]
