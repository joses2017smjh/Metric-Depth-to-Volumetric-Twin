"""Phase 0 verification: loaders parse the real release files correctly.

Run from the repo root with the downloaded data in data/:
    pytest tests/test_phase0.py -v
"""

from pathlib import Path

import numpy as np
import pytest

from v3d import (
    Camera,
    ISSIA_FLIPPED_CAMERAS,
    load_issia_calibration,
    load_issia_csv,
    load_labels_v3d,
    load_snv3d_csv,
    load_split,
)
from v3d.labels import iter_matches
from v3d.snv3d import split_frames

DATA = Path(__file__).resolve().parents[1] / "data"

pytestmark = pytest.mark.skipif(
    not (DATA / "annotations" / "SNv3D.csv").exists(),
    reason="release data not downloaded into data/",
)


@pytest.fixture(scope="module")
def snv3d():
    return load_snv3d_csv(DATA / "annotations" / "SNv3D.csv")


def test_snv3d_parses(snv3d):
    assert len(snv3d) > 4000
    row = snv3d[snv3d["ball_3D"].notna() & snv3d["ball_bbox"].notna()].iloc[0]
    assert isinstance(row["calibration"], dict)
    assert row["ball_bbox"].shape == (4,)
    assert row["ball_3D"].shape == (3,)
    # Ball must be on the pitch (105 x 68 m, origin at center) and near the
    # ground plane; |z| small regardless of which way z points.
    x, y, z = row["ball_3D"]
    assert abs(x) < 60 and abs(y) < 40 and abs(z) < 5


def test_splits_match_csv(snv3d):
    train = load_split(DATA / "splits" / "SNv3D-train.txt")
    test = load_split(DATA / "splits" / "SNv3D-test.txt")
    assert not set(train) & set(test), "train/test keys overlap"
    covered = set(snv3d["frame_id"])
    # Every split key should identify a row in the CSV.
    assert set(train) <= covered
    assert set(test) <= covered


def test_camera_reprojection_matches_published_error(snv3d):
    """Project ball_3D with our Camera and compare against rep_error.

    This validates the whole geometry stack: parsing, K/R/t assembly, and
    projection convention. We check the median over annotated action frames
    rather than exact per-row equality because rep_error was computed
    against the annotated 2D ball center, which we reconstruct from the
    bbox: center = (x + w/2, y + h/2).
    """
    rows = snv3d[
        snv3d["ball_3D"].notna()
        & snv3d["ball_bbox"].notna()
        & snv3d["calibration"].notna()
        & snv3d["rep_error"].notna()
    ]
    assert len(rows) > 100
    diffs = []
    for _, r in rows.head(500).iterrows():
        cam = Camera.from_soccernet(r["calibration"])
        pixel = cam.project(r["ball_3D"])[0]
        bx, by, bw, bh = r["ball_bbox"]
        center = np.array([bx + bw / 2, by + bh / 2])
        our_err = float(np.linalg.norm(pixel - center))
        diffs.append(abs(our_err - float(r["rep_error"])))
    median_diff = float(np.median(diffs))
    assert median_diff < 1.0, f"median |our_err - rep_error| = {median_diff:.3f}px"


def test_labels_v3d_loads_and_links():
    labels_root = DATA / "v3d-labels"
    if not labels_root.exists():
        pytest.skip("v3d-labels not extracted")
    match = next(iter_matches(labels_root))
    assert match.actions and match.replays
    groups = match.multiview_groups()
    assert groups, "no action->calibrated_replay links found"
    action, replays = groups[0]
    assert action.camera() is not None
    assert all(r.camera() is not None for r in replays)
    assert set(action.jac) == {"JaC@0.005", "JaC@0.01", "JaC@0.02"}
    # Player bboxes carry pose keypoints [x, y, confidence].
    boxed = [b for b in action.bboxes if b.get("keypoints")]
    if boxed:
        assert len(boxed[0]["keypoints"][0]) == 3


def test_issia_loaders():
    cams = load_issia_calibration(DATA / "annotations" / "issia_calibration.json")
    assert set(cams) == {1, 2, 3, 4, 5, 6}
    for cam in cams.values():
        assert cam.K.shape == (3, 3)
        np.testing.assert_allclose(cam.t, -cam.R @ cam.position)

    df = load_issia_csv(DATA / "annotations" / "ISSIA-3D.csv")
    assert len(df) > 1000
    tri = df[df["ball_3D"].notna()]
    assert len(tri) > 0
    assert tri.iloc[0]["ball_3D"].shape == (3,)

    # Triangulated balls should land on the pitch for ISSIA too.
    sample = np.stack(tri["ball_3D"].head(200).to_numpy())
    assert np.nanmedian(np.abs(sample[:, 0])) < 60
    assert np.nanmedian(np.abs(sample[:, 1])) < 40


def test_issia_flip_roundtrip():
    from v3d.issia import flip_x, unflip_image

    img = np.arange(24).reshape(2, 4, 3)
    assert np.array_equal(unflip_image(img, 1), img)
    assert np.array_equal(unflip_image(img, 2), img[:, ::-1])
    w = 1920
    for c in ISSIA_FLIPPED_CAMERAS:
        assert flip_x(flip_x(100.0, w, c), w, c) == 100.0
    assert flip_x(100.0, w, 3) == 100.0
