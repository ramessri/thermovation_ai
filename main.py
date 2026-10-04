"""
main.py — batch-run pipe extraction (segmentation) + placement over the
already-reconstructed videos in sfm 1 (calibrated batch run).

Avoids PowerShell array/quoting issues by doing the loop in Python instead
of shell script. Each stage is a plain subprocess call to the existing
07_pipes/pipe_paths.py and 06_placement/placement_3d.py scripts.

Usage:
  python main.py                 # gdino pipe extraction + placement, all 24 videos
  python main.py --yoloe         # also run the yoloe second-opinion pass
  python main.py --only IMG_3126 # just one video (for testing)
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
PY = sys.executable

OUT = Path(r"C:\thermovation-output\sfm 1")

# video work-dir name -> sparse model index (best_sparse_model from each report.json)
MODEL = {
    "Albert_Mayer_IMG_1831": 3,
    "Andreas__Scholz__VID_20260802_142216161": 1,
    "Bjoern-Harald_Malluche_Heizraum_Video": 0,
    "Christian__Heimes_IMG_4716": 0,
    "Dietmar_Baudisch_VID_20260731_142741": 0,
    "Dominik__Lindhorst__PXL_20260731_124618342": 1,
    "Friedhelm_Bednarz_VIDEO-2026-08-06-17-56-31": 0,
    "Helmut_Schlierf": 0,
    "IMG_3126": 0,
    "IMG_3128": 0,
    "Johannes_Steinhauser_VID_20260804_192842": 1,
    "Klaus_Rombergg": 0,
    "Leif_Malluche_20260802_115810": 0,
    "Manfred_Hahn_IMG_2663": 4,
    "Michael_Speth_IMG_4149": 0,
    "Monika_Adldinger_20260801_174346": 0,
    "Monika__Mulock_IMG_6278": 0,
    "Moritz__Schneider_IMG_7306": 0,
    "Renate_Hefele_20260802_180013": 0,
    "Ulli_Roessle_Heizkeller_Lechermann": 0,
    "WhatsApp_Video_2026-07-05_at_17.41.45": 1,
    "WhatsApp_Video_2026-07-05_at_17.41.49": 0,
    "WhatsApp_Video_2026-07-09_at_09.07.37": 1,
    "Wolfram_Koestler_VID-20260801-WA0002": 0,
}


def run(cmd: list, label: str) -> bool:
    cmd = [str(c) for c in cmd]
    print(f"\n=== {label} ===\n$ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=ROOT)
    if res.returncode != 0:
        print(f"  [{label}] FAILED (exit {res.returncode}) — continuing with next video")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None,
                        help="Restrict to a single video name (key in MODEL) for testing")
    parser.add_argument("--yoloe", action="store_true",
                        help="Also run the yoloe second-opinion pipe extraction pass")
    parser.add_argument("--skip-placement", action="store_true")
    args = parser.parse_args()

    videos = [args.only] if args.only else list(MODEL.keys())

    # Step 1: gdino pipe extraction
    for video in videos:
        model = MODEL[video]
        sfm_dir = OUT / video / "sfm"
        frames_dir = OUT / video / "frames"
        run([PY, "07_pipes/pipe_paths.py", sfm_dir, frames_dir,
             "--model", model, "--detector", "gdino"], f"{video} [gdino]")

    # Step 2: placement (reads pipe_lengths.json written by step 1)
    if not args.skip_placement:
        for video in videos:
            model = MODEL[video]
            sfm_dir = OUT / video / "sfm"
            frames_dir = OUT / video / "frames"
            run([PY, "06_placement/placement_3d.py", sfm_dir, frames_dir,
                 "--model", model], f"{video} [placement]")

    # Step 3 (optional): yoloe second opinion, renamed so it doesn't clobber
    # the gdino result placement already used
    if args.yoloe:
        for video in videos:
            model = MODEL[video]
            sfm_dir = OUT / video / "sfm"
            frames_dir = OUT / video / "frames"
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
