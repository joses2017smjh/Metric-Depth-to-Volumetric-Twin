"""Loader for SNv3D.csv — the flat per-frame annotation table.

One row per frame (action or replay) with a ball annotation and/or
calibration. Columns of note (see the SoccerNet-v3D README):

* ball_bbox        [x, y, w, h] pixels (top-left corner + size)
* calibration      SoccerNet-calibration dict (parsed to dict here)
* JaC@*            calibration quality at three tolerances
* ball_3D          triangulated 3D ball position, meters, pitch frame
* rep_error        reprojection error of ball_3D into this frame, pixels
* optimized_error  projection error after bbox optimization, meters
* optimized_d      optimized ball diameter in image space, pixels
* set              "train"/"test" — the published ball-detector split

Frame identity: (match, main_action, replay). Action frames have
action == True and replay == NaN; replays carry the replay index. The
split .txt files identify frames as "match/main_action" for actions and
"match/main_action_replay" for replays — `frame_id()` reproduces that key.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from v3d.parsing import literal, parse_ball_3d, parse_bbox_xywh


def load_snv3d_csv(path: str | Path) -> pd.DataFrame:
    """Load SNv3D.csv with the repr-string columns parsed into objects.

    After loading: `calibration` holds dicts (or None), `ball_bbox` holds
    (4,) float arrays [x, y, w, h] (or None), `ball_3D` holds (3,) float
    arrays in meters (or None), and `frame_id` matches the split-file keys.
    """
    df = pd.read_csv(path, index_col=0)

    df["calibration"] = df["calibration"].map(
        lambda s: literal(s) if isinstance(s, str) else None
    )
    df["ball_bbox"] = df["ball_bbox"].map(parse_bbox_xywh)
    df["ball_3D"] = df["ball_3D"].map(parse_ball_3d)

    replay = df["replay"].astype("Int64")
    action_key = df["match"] + "/" + df["main_action"].astype(str)
    df["frame_id"] = action_key.where(
        replay.isna(), action_key + "_" + replay.astype(str)
    )
    return df


def load_split(path: str | Path) -> list[str]:
    """Load SNv3D-train.txt / SNv3D-test.txt: one 'match/frame' key per line."""
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def split_frames(df: pd.DataFrame, split_keys: list[str]) -> pd.DataFrame:
    """Rows of df whose frame_id appears in a split file's keys."""
    keys = set(split_keys)
    return df[df["frame_id"].isin(keys)]
