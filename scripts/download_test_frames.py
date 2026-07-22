"""Fetch SNv3D-test frames (61 games), keeping only split-referenced frames.

Run from repo root:  python scripts/download_test_frames.py
"""

import sys
from pathlib import Path

from v3d.download import download_split_frames

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main():
    counts = download_split_frames(
        split_file=DATA / "splits" / "SNv3D-test.txt",
        games_file=DATA / "frame_lists" / "test_games.txt",
        soccernet_dir=DATA / "SoccerNet",
        frames_out=DATA / "frames" / "test",
        spl="test",
    )
    total = sum(counts.values())
    print(f"\nextracted {total} frames across {len(counts)} games")
    zero = [m for m, n in counts.items() if n == 0]
    if zero:
        print(f"WARNING: {len(zero)} games yielded 0 frames:")
        for m in zero[:10]:
            print("  ", m)
    sys.exit(1 if zero else 0)


if __name__ == "__main__":
    main()
