"""
run_sfm_disk_lightglue.py — SfM reconstruction using DISK (learned feature
detector/descriptor) + LightGlue (learned matcher) instead of this project's
default SIFT + sequential-matching pipeline (03_reconstruction/run_sfm.py).

Why: this session spent most of its effort on placement_3d.py's
--min-wall-inliers evidence floor, and root-caused most of its refusals to
a real, confirmed COLMAP limitation — SIFT is a corner/blob detector, and
plain painted walls carry almost no SIFT-matchable texture, so the sparse
cloud is structurally starved specifically ON THE WALL, independent of
video quality (see 06_placement/placement_3d.py's module comments /
context.txt session log). DISK and LightGlue are both LEARNED (trained end-
to-end on real scenes, not hand-crafted corner detection), and are reported
in the literature to hold up substantially better on low-texture surfaces —
this script tests that directly on this project's own hardest cases instead
of taking the claim on faith.

Uses hloc (Hierarchical-Localization, github.com/cvg/Hierarchical-
Localization) rather than hand-writing pycolmap Database import calls —
hloc already implements DISK extraction -> LightGlue matching -> COLMAP
database import -> pycolmap incremental mapping correctly; re-deriving that
custom-feature-import logic from scratch under time pressure was assessed
as a real correctness risk not worth taking for a comparison experiment.

hloc ships pair generators for exhaustive/retrieval/covisibility/pose-based
matching, but NOT sequential-window (this project's own SIFT pipeline uses
sequential matching, appropriate for a continuous video walkthrough) — so
pair generation is done directly here instead (a plain "img1.jpg img2.jpg"
text file, which is all reconstruction.main() actually needs).

No marker-based metric scale is computed here (out of scope for this
comparison) — output is registered-image count and raw point density,
directly comparable to this project's own existing SIFT-based sfm 1
reconstructions on the same videos without needing scale.json at all.

Usage:
  python 03_reconstruction/run_sfm_disk_lightglue.py \
      --frames-dir "C:\\thermovation-output\\sfm 1\\Klaus_Rombergg\\frames" \
      --out output/disk_lightglue/Klaus_Rombergg --window 15
"""

import argparse
import json
import os
import time
from pathlib import Path


def sanitize_frames_dir(frames_dir: Path, work_dir: Path) -> Path:
    """hloc's pairs.txt parser (parse_retrieval) splits each line on ANY
    whitespace expecting exactly 2 tokens — this project's own source video
    filenames carry embedded spaces (e.g. 'Bjoern-Harald_Malluche_Heizraum
    Video_0066.jpg', confirmed all session), which breaks that unpacking.
    Hardlink (same volume, no extra disk space) frames into a space-free
    staging dir rather than patching hloc's parser."""
    dst = work_dir / "frames_sanitized"
    dst.mkdir(parents=True, exist_ok=True)
    for src in sorted(frames_dir.glob("*.jpg")):
        clean_name = src.name.replace(" ", "_")
        dst_path = dst / clean_name
        if not dst_path.exists():
            try:
                os.link(src, dst_path)
            except OSError:
                dst_path.write_bytes(src.read_bytes())
    return dst


def make_sequential_pairs(image_names: list[str], window: int) -> list[tuple[str, str]]:
    pairs = []
    n = len(image_names)
    for i in range(n):
        for j in range(i + 1, min(i + 1 + window, n)):
            pairs.append((image_names[i], image_names[j]))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="SfM via DISK + LightGlue instead of SIFT")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window", type=int, default=15,
                        help="Sequential-pairing window (consecutive frames each frame is "
                             "matched against) — mirrors this project's SIFT pipeline's "
                             "sequential-matching strategy, tractable for large frame counts "
                             "unlike hloc's built-in exhaustive pairing.")
    parser.add_argument("--max-keypoints", type=int, default=4096)
    args = parser.parse_args()

    from hloc import extract_features, match_features, reconstruction

    args.out.mkdir(parents=True, exist_ok=True)
    frames_dir = sanitize_frames_dir(args.frames_dir, args.out)
    image_names = sorted(p.name for p in frames_dir.glob("*.jpg"))
    if not image_names:
        print(f"FAILED: no .jpg frames found in {args.frames_dir}")
        return
    print(f"{len(image_names)} frames in {args.frames_dir} (staged space-free at {frames_dir})")

    pairs_path = args.out / "pairs.txt"
    pairs = make_sequential_pairs(image_names, args.window)
    with open(pairs_path, "w") as f:
        for a, b in pairs:
            f.write(f"{a} {b}\n")
    print(f"Sequential pairs (window={args.window}): {len(pairs)} pairs -> {pairs_path}")

    disk_conf = extract_features.confs["disk"]
    disk_conf["model"]["max_keypoints"] = args.max_keypoints
    lg_conf = match_features.confs["disk+lightglue"]

    t0 = time.time()
    feature_path = extract_features.main(disk_conf, frames_dir, args.out)
    t_extract = time.time() - t0

    t0 = time.time()
    match_path = match_features.main(lg_conf, pairs_path, feature_path.stem, args.out)
    t_match = time.time() - t0

    sfm_dir = args.out / "sparse"
    t0 = time.time()
    rec = reconstruction.main(sfm_dir, frames_dir, pairs_path, feature_path, match_path)
    t_map = time.time() - t0

    result = dict(
        frames_dir=str(args.frames_dir),
        n_frames=len(image_names),
        window=args.window,
        n_pairs=len(pairs),
        n_registered=rec.num_reg_images() if rec else 0,
        n_points3D=len(rec.points3D) if rec else 0,
        extract_s=round(t_extract, 1),
        match_s=round(t_match, 1),
        map_s=round(t_map, 1),
    )
    (args.out / "disk_lightglue_result.json").write_text(json.dumps(result, indent=2))
    print(f"Registered: {result['n_registered']}/{result['n_frames']}  "
          f"Points3D: {result['n_points3D']}  "
          f"(extract {t_extract:.0f}s, match {t_match:.0f}s, map {t_map:.0f}s)")
    print(f"Saved {args.out / 'disk_lightglue_result.json'}")


if __name__ == "__main__":
    main()
