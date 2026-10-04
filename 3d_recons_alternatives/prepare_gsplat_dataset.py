"""
prepare_gsplat_dataset.py — arrange an existing SfM run's output into the
folder layout gsplat's own COLMAP dataset loader expects, so 3D Gaussian
Splatting reuses the same camera poses, points, and marker scale as the SfM
arm instead of re-running feature matching from scratch.

gsplat's Parser (examples/datasets/colmap.py in the gsplat repo) loads via
`pycolmap.Reconstruction(colmap_dir)` — the exact library this repo already
uses everywhere — and undistorts images itself at load time (cv2.remap from
the COLMAP camera's own distortion params), so no separate undistortion
step is needed here; this script just copies files into place.

Usage:
  python 3d_recons_alternatives/prepare_gsplat_dataset.py \
      --sfm-dir output/sfm/IMG_3126 --frames-dir dataset/frames/IMG_3126_sfm \
      --model 1 --out 3d_recons_alternatives/data/IMG_3126
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare a gsplat-ready COLMAP dataset")
    parser.add_argument("--sfm-dir", type=Path, required=True,
                        help="Existing SfM output dir (contains sparse/<model>/, scale.json)")
    parser.add_argument("--frames-dir", type=Path, required=True,
                        help="The same frames dir used for this SfM run")
    parser.add_argument("--model", default="1", help="Sparse model index (default 1)")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output dataset dir for gsplat, e.g. 3d_recons_alternatives/data/<video>")
    args = parser.parse_args()

    src_sparse = args.sfm_dir / "sparse" / args.model
    if not src_sparse.exists():
        raise SystemExit(f"No sparse model at {src_sparse}")
    if not args.frames_dir.exists():
        raise SystemExit(f"No frames dir at {args.frames_dir}")

    dst_sparse = args.out / "sparse" / "0"
    dst_images = args.out / "images"
    dst_sparse.mkdir(parents=True, exist_ok=True)
    dst_images.mkdir(parents=True, exist_ok=True)

    for f in src_sparse.iterdir():
        shutil.copy2(f, dst_sparse / f.name)
    print(f"Copied sparse model: {src_sparse} -> {dst_sparse}")

    n = 0
    for f in sorted(args.frames_dir.glob("*.jpg")):
        shutil.copy2(f, dst_images / f.name)
        n += 1
    print(f"Copied {n} frames: {args.frames_dir} -> {dst_images}")

    # carry the marker scale along so gaussians_to_room_dims.py doesn't need
    # to know where the original SfM run lived
    scale_path = args.sfm_dir / "scale.json"
    if scale_path.exists():
        shutil.copy2(scale_path, args.out / "scale.json")
        print(f"Copied scale.json -> {args.out / 'scale.json'}")
    else:
        print(f"WARNING: no scale.json at {scale_path} — gaussians_to_room_dims.py "
              "will need --cm-per-unit passed manually")

    print(f"\nReady: {args.out}")
    print("Train with gsplat's own trainer — see 3d_recons_alternatives/README.md for the exact command.")


if __name__ == "__main__":
    main()
