"""
main.py — batch-run pipe extraction (segmentation) + placement over every
already-reconstructed video under OUTPUT_DIR (or --root).

Each video folder needs sfm/ and frames/ (as written by run.py). The sparse
model with the most registered images is used, same as run.py.

Usage:
  python main.py                 # gdino pipe extraction + placement, all videos
  python main.py --yoloe         # also run the yoloe second-opinion pass
  python main.py --only IMG_3126 # just one video (for testing)
  python main.py --root "output/pipeline"
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from run import best_sparse_model

ROOT = Path(__file__).parent
PY = sys.executable

load_dotenv(ROOT / ".env")
DEFAULT_ROOT = Path(os.environ.get("OUTPUT_DIR", "output/pipeline"))


def run(cmd: list, label: str) -> bool:
    cmd = [str(c) for c in cmd]
    print(f"\n=== {label} ===\n$ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=ROOT)
    if res.returncode != 0:
        print(f"  [{label}] FAILED (exit {res.returncode}) — continuing with next video")
        return False
    return True


def find_videos(root: Path, only: str | None) -> dict[str, int]:
    """Map video folder name -> best sparse model index."""
    videos = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir() or (only and d.name != only):
            continue
        model_dir = best_sparse_model(d / "sfm")
        if model_dir is None or not (d / "frames").exists():
            print(f"  skipping {d.name}: no sparse model or frames")
            continue
        videos[d.name] = int(model_dir.name)
    return videos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help=f"Folder of per-video run.py outputs (default: {DEFAULT_ROOT})")
    parser.add_argument("--only", default=None,
                        help="Restrict to a single video folder name for testing")
    parser.add_argument("--yoloe", action="store_true",
                        help="Also run the yoloe second-opinion pipe extraction pass")
    parser.add_argument("--skip-placement", action="store_true")
    args = parser.parse_args()

    videos = find_videos(args.root, args.only)

    # Step 1: gdino pipe extraction
    for video, model in videos.items():
        sfm_dir = args.root / video / "sfm"
        frames_dir = args.root / video / "frames"
        run([PY, "07_pipes/pipe_paths.py", sfm_dir, frames_dir,
             "--model", model, "--detector", "gdino"], f"{video} [gdino]")

    # Step 2: placement (reads pipe_lengths.json written by step 1)
    if not args.skip_placement:
        for video, model in videos.items():
            sfm_dir = args.root / video / "sfm"
            frames_dir = args.root / video / "frames"
            run([PY, "06_placement/placement_3d.py", sfm_dir, frames_dir,
                 "--model", model], f"{video} [placement]")

    # Step 3 (optional): yoloe second opinion, renamed so it doesn't clobber
    # the gdino result placement already used
    if args.yoloe:
        for video, model in videos.items():
            sfm_dir = args.root / video / "sfm"
            frames_dir = args.root / video / "frames"
            ok = run([PY, "07_pipes/pipe_paths.py", sfm_dir, frames_dir,
                     "--model", model, "--detector", "yoloe",
                     "--pipe-prompt", "heating pipe", "--threshold", "0.05"],
                     f"{video} [yoloe]")
            if ok:
                pl = sfm_dir / "pipe_lengths.json"
                pj = sfm_dir / "pipe_paths.jpg"
                if pl.exists():
                    shutil.move(pl, sfm_dir / "pipe_lengths_yoloe.json")
                if pj.exists():
                    shutil.move(pj, sfm_dir / "pipe_paths_yoloe.jpg")


if __name__ == "__main__":
    main()
