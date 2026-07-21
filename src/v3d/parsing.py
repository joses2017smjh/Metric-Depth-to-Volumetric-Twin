"""Parsers for the quirky serialization in the SoccerNet-v3D release files.

The release stores Python reprs rather than clean JSON in several places:

* Labels-v3D.json is a JSON *string* whose content is a Python dict repr
  (single quotes), so json.load gives a str that json.loads can't decode.
* The SNv3D.csv `calibration` column is a single-quoted Python dict literal.
* The `ball_3D` column is literally "[array([x, y, z])]" — a repr of a list
  holding a numpy array.

`literal()` handles all three with one restricted eval: no builtins, and the
only names resolvable are `array` (mapped to np.array) and `nan`. These are
trusted research files from the paper's GitHub release; do not point this at
untrusted input.
"""

from __future__ import annotations

import numpy as np

_EVAL_GLOBALS = {"__builtins__": {}}
_EVAL_LOCALS = {"array": np.array, "nan": float("nan")}


def literal(s: str):
    """Parse a Python-repr string from the v3D release files."""
    return eval(s, _EVAL_GLOBALS, _EVAL_LOCALS)  # noqa: S307 — trusted data, see module docstring


def parse_ball_3d(s) -> np.ndarray | None:
    """Parse a ball_3D column value -> (3,) meters, or None.

    SNv3D.csv stores "[array([x, y, z])]" (a list wrapping a numpy array);
    ISSIA-3D.csv stores a plain "[x, y, z]". Both parse here.
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    if isinstance(s, np.ndarray):
        return s.reshape(3)
    v = literal(s)
    if isinstance(v, list) and len(v) == 1 and isinstance(v[0], (list, np.ndarray)):
        v = v[0]
    return np.asarray(v, dtype=float).reshape(3)


def parse_bbox_xywh(s) -> np.ndarray | None:
    """Parse the ball_bbox column "[x, y, w, h]" -> (4,) pixels, or None."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    v = np.asarray(literal(s), dtype=float).reshape(4)
    return v
