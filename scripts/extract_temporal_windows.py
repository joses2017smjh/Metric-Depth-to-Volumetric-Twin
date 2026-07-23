"""Track 1: fetch temporal frame windows around SNv3D-test action keyframes.

Each annotated ACTION frame carries `half` (which video) and `position`
(milliseconds into that video); validated to align with the released 720p
video to within one frame (NCC ~0.995 at offset 0). Replay frames store the
*action's* timestamp rather than their own airtime, so only action frames get
usable temporal context.

Data economics: a 720p half is ~1 GB and yields only ~3 annotated frames, so we
stream one video at a time — download, extract the +/-W frame windows, delete
the video. Peak disk stays ~1 GB regardless of how many matches are processed.

CREDENTIALS: the SoccerNet videos are NDA-gated. Pass the password via the
SOCCERNET_PASSWORD environment variable. It is deliberately never written to
disk or committed to the repository.

Usage:
    export SOCCERNET_PASSWORD=...
    python scripts/extract_temporal_windows.py --max-files 40 --half-window 4
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import urllib.parse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

VIDEO_720P_USER = "xNGfp1W3wPeVOmQ"  # public share id for the 720p bucket
FPS = 25.0
FRAME_MS = 1000.0 / FPS


def owncloud_server() -> str:
    from SoccerNet.Downloader import SoccerNetDownloader

    return SoccerNetDownloader(LocalDirectory="/tmp/_sn").OwnCloudServer


def download_video(server: str, game: str, half: int, dest: Path, password: str) -> bool:
    """Fetch one 720p half with curl (retries + resume; urlretrieve truncates)."""
    fname = f"{half}_720p.mkv"
    url = f"{server}/video_720p/{urllib.parse.quote(game)}/{fname}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl", "-sS", "-f", "--retry", "8", "--retry-all-errors",
        "--retry-delay", "3", "-C", "-",
        "--user", f"{VIDEO_720P_USER}:{password}",
        "-o", str(dest), url,
    ]
    return subprocess.run(cmd).returncode == 0


def extract_windows(video: Path, jobs: pd.DataFrame, out_root: Path,
                    half_window: int, upscale: tuple[int, int] | None) -> int:
    """Extract +/-half_window frames around each job's position.

    Frames are named <frame_stem>_<k>.png with k in [-W..W]; k=0 is the
    annotated keyframe's own timestamp.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return 0
    written = 0
    for _, r in jobs.iterrows():
        stem = str(r["frame"]).removesuffix(".png")
        dest = out_root / r["match"]
        dest.mkdir(parents=True, exist_ok=True)
        # Seek once to the start of the window, then read sequentially — much
        # faster and more reliable than seeking per frame.
        start_ms = r["pos"] - half_window * FRAME_MS
        cap.set(cv2.CAP_PROP_POS_MSEC, max(start_ms, 0))
        for k in range(-half_window, half_window + 1):
            ok, frame = cap.read()
            if not ok:
                break
            out = dest / f"{stem}_{k:+d}.png"
            if upscale is not None:
                frame = cv2.resize(frame, upscale, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(out), frame)
            written += 1
    cap.release()
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-csv", default=str(DATA / "frame_lists" / "test_action_frames.csv"))
    ap.add_argument("--out", default=str(DATA / "frames" / "test_windows"))
    ap.add_argument("--video-dir", default=str(DATA / "video"))
    ap.add_argument("--half-window", type=int, default=4, help="+/-N frames around the keyframe")
    ap.add_argument("--max-files", type=int, default=40,
                    help="how many (match, half) videos to process, densest first")
    ap.add_argument("--upscale", default="1920x1080",
                    help="resize extracted frames to match annotation space; 'none' to keep 720p")
    ap.add_argument("--keep-video", action="store_true")
    args = ap.parse_args()

    password = os.environ.get("SOCCERNET_PASSWORD")
    if not password:
        raise SystemExit("set SOCCERNET_PASSWORD (NDA-gated videos); never commit it")

    up = None
    if args.upscale.lower() != "none":
        w, h = args.upscale.lower().split("x")
        up = (int(w), int(h))

    d = pd.read_csv(args.frames_csv)
    # Densest video files first: most annotated frames per GB downloaded.
    order = (d.groupby(["game", "half"]).size().sort_values(ascending=False)
             .reset_index(name="n").head(args.max_files))

    server = owncloud_server()
    out_root = Path(args.out)
    video_dir = Path(args.video_dir)
    total_frames = 0
    done_files = 0

    for _, row in order.iterrows():
        game, half, n = row["game"], int(row["half"]), int(row["n"])
        jobs = d[(d.game == game) & (d.half == half)]
        match = jobs["match"].iloc[0]
        # Skip if every window is already extracted.
        need = False
        for _, j in jobs.iterrows():
            stem = str(j["frame"]).removesuffix(".png")
            if not (out_root / match / f"{stem}_{0:+d}.png").exists():
                need = True
                break
        if not need:
            done_files += 1
            continue

        vid = video_dir / game / f"{half}_720p.mkv"
        if not vid.exists():
            print(f"[{done_files+1}/{len(order)}] downloading {match} half {half} ({n} frames)...",
                  flush=True)
            if not download_video(server, game, half, vid, password):
                print(f"   FAILED download: {game} half {half}", flush=True)
                continue
        w = extract_windows(vid, jobs, out_root, args.half_window, up)
        total_frames += w
        done_files += 1
        print(f"   extracted {w} frames ({n} keyframes) -> {match}", flush=True)
        if not args.keep_video:
            vid.unlink(missing_ok=True)
            # prune empty dirs
            for p in (vid.parent, vid.parent.parent):
                try:
                    p.rmdir()
                except OSError:
                    pass

    print(f"\nprocessed {done_files} video files; extracted {total_frames} frames -> {out_root}")


if __name__ == "__main__":
    main()
