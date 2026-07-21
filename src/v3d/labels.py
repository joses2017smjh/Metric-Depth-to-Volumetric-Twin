"""Loader for per-match Labels-v3D.json files.

One Labels-v3D.json per match, following the SoccerNet-v3 folder layout
(league/season/match/Labels-v3D.json). Each file holds:

* GameMetadata: match URL, counts, frame lists, `reconstructions`
  (empty dicts in release v1.0.0 — the action->calibrated_replays links
  below are the usable multi-view structure).
* actions:  {"<action>.png": frame}  — main-camera shots.
* replays:  {"<action>_<k>.png": frame} — replay shots of the same moment.

Each frame carries:
* imageMetadata — width/height, gameTime, and for actions the key field
  `calibrated_replays`: the replay frames whose calibration passed the
  JaC@0.005 >= 0.75 quality gate, i.e. the other views you can triangulate
  against.
* calibration — SoccerNet-calibration dict (see v3d.calibration.Camera).
* JaC — {"JaC@0.005", "JaC@0.01", "JaC@0.02"}: Jaccard index between the
  pitch-line projection under the estimated calibration and the annotated
  lines; higher = better calibration, evaluated at three pixel tolerances
  (fractions of the image diagonal).
* bboxes — object annotations (players, referees, goalkeepers) with class,
  box corners, and 17-point pose keypoints [x, y, confidence] — the raw
  material for the player-lifting phase.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from v3d.calibration import Camera
from v3d.parsing import literal


@dataclass
class Frame:
    """One annotated frame (action or replay) from Labels-v3D.json."""

    name: str  # e.g. "11.png" (action) or "11_0.png" (replay)
    is_action: bool
    metadata: dict
    calibration: dict | None
    jac: dict  # {"JaC@0.005": float, ...}
    bboxes: list[dict]

    @property
    def calibrated_replays(self) -> list[str]:
        """Replay frames calibrated well enough to triangulate against (actions only)."""
        return self.metadata.get("calibrated_replays", []) or []

    def camera(self) -> Camera | None:
        return Camera.from_soccernet(self.calibration) if self.calibration else None


@dataclass
class MatchLabels:
    """All v3D annotations for one match."""

    path: Path
    metadata: dict
    actions: dict[str, Frame]
    replays: dict[str, Frame]

    @property
    def match_name(self) -> str:
        return self.path.parent.name

    def multiview_groups(self) -> list[tuple[Frame, list[Frame]]]:
        """(action, calibrated replay frames) pairs with >= 1 usable replay.

        These are the triangulation units of the dataset: the same instant
        seen from the main camera plus replay cameras.
        """
        groups = []
        for frame in self.actions.values():
            reps = [self.replays[r] for r in frame.calibrated_replays if r in self.replays]
            if reps:
                groups.append((frame, reps))
        return groups


def _to_frame(name: str, raw: dict, is_action: bool) -> Frame:
    return Frame(
        name=name,
        is_action=is_action,
        metadata=raw.get("imageMetadata", {}),
        calibration=raw.get("calibration"),
        jac=raw.get("JaC", {}),
        bboxes=raw.get("bboxes", []),
    )


def load_labels_v3d(path: str | Path) -> MatchLabels:
    """Load one Labels-v3D.json.

    The file is double-encoded: a JSON string whose content is a Python dict
    repr — hence json.load followed by the restricted literal() parser.
    """
    path = Path(path)
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, str):
        raw = literal(raw)

    return MatchLabels(
        path=path,
        metadata=raw.get("GameMetadata", {}),
        actions={k: _to_frame(k, v, True) for k, v in raw.get("actions", {}).items()},
        replays={k: _to_frame(k, v, False) for k, v in raw.get("replays", {}).items()},
    )


def iter_matches(root: str | Path):
    """Yield MatchLabels for every Labels-v3D.json under root (league/season/match)."""
    for p in sorted(Path(root).glob("*/*/*/Labels-v3D.json")):
        yield load_labels_v3d(p)
