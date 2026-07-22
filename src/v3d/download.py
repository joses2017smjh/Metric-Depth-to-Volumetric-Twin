"""Download SoccerNet-v3 frames for the SNv3D games, keeping only what we need.

Each game's Frames-v3.zip is ~300 MB and holds every v3 frame for the match,
but the SNv3D split references only ~13 of them per game. To stay light on
disk we download a zip, extract just the frames named in the split, then
delete the zip. Net footprint is a few MB per game instead of 300.

The frames are NOT NDA-gated: the SoccerNet package ships the shared-folder
credentials for Frames-v3.zip (password "SoccerNet_Reviewers_SDATA"), so no
form or manual password is required.

Frame naming: a split key "<match>/<action>" maps to "<action>.png" (a main
camera frame) and "<match>/<action>_<k>" to "<action>_<k>.png" (a replay),
which is exactly how frames are named inside the zip.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from SoccerNet.Downloader import SoccerNetDownloader

from v3d.snv3d import load_split


def frames_for_game(split_keys: list[str], game_match_name: str) -> set[str]:
    """Frame file names ("15.png", "19_1.png") referenced for one match."""
    wanted = set()
    for key in split_keys:
        match, frame = key.rsplit("/", 1)
        if match == game_match_name:
            wanted.add(frame + ".png")
    return wanted


def download_and_prune(
    games: list[str],
    split_keys: list[str],
    soccernet_dir: str | Path,
    frames_out: str | Path,
    spl: str = "test",
    keep_zip: bool = False,
) -> dict[str, int]:
    """Download Frames-v3.zip per game, extract split frames, prune the zip.

    Parameters
    ----------
    games        league/season/match paths (as SoccerNet expects).
    split_keys   keys from a SNv3D split file (match/frame).
    soccernet_dir  scratch dir where SoccerNet writes the zips.
    frames_out   destination root; frames land in frames_out/<match>/<frame>.png.
    spl          SoccerNet split label for downloadGame (cosmetic here).
    keep_zip     keep the downloaded zip instead of deleting it.

    Returns {match_name: n_frames_extracted}.
    """
    soccernet_dir = Path(soccernet_dir)
    frames_out = Path(frames_out)
    downloader = SoccerNetDownloader(LocalDirectory=str(soccernet_dir))

    counts: dict[str, int] = {}
    for game in games:
        match_name = Path(game).name
        wanted = frames_for_game(split_keys, match_name)
        dest = frames_out / match_name
        # Skip if already fully extracted.
        if wanted and all((dest / f).exists() for f in wanted):
            counts[match_name] = len(wanted)
            continue

        zip_path = soccernet_dir / game / "Frames-v3.zip"
        if not zip_path.exists():
            downloader.downloadGame(game=game, files=["Frames-v3.zip"], spl=spl, verbose=False)

        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
            for f in wanted:
                if f in names:
                    with zf.open(f) as src, open(dest / f, "wb") as out:
                        out.write(src.read())
                    n += 1
        counts[match_name] = n
        if not keep_zip:
            zip_path.unlink(missing_ok=True)

    return counts


def download_split_frames(
    split_file: str | Path,
    games_file: str | Path,
    soccernet_dir: str | Path,
    frames_out: str | Path,
    spl: str = "test",
) -> dict[str, int]:
    """Convenience wrapper: read a split + games list and fetch their frames."""
    split_keys = load_split(split_file)
    games = [g.strip() for g in Path(games_file).read_text().splitlines() if g.strip()]
    return download_and_prune(games, split_keys, soccernet_dir, frames_out, spl=spl)
