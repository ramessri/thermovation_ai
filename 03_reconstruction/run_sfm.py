"""
run_sfm.py — Structure-from-Motion baseline via pycolmap.

Pipeline: SIFT feature extraction → sequential matching (video ordering)
→ incremental mapping. Produces a sparse COLMAP model that is later scaled
to metric units with the fiducial marker (see detect_marker.py).

Camera intrinsics: by default COLMAP self-calibrates focal length as a free
parameter during bundle adjustment, which is a known source of systematic
scale bias. Pass --calib with a calibrate_camera.py output to fix intrinsics
to a value measured directly from the marker (Zhang's method) instead.

Usage:
  python 03_reconstruction/run_sfm.py dataset/frames/<video_(sfm)> --output output/sfm/video
  python 03_reconstruction/run_sfm.py <frames_dir> --output <out> --calib output/calib/video.json --gpu
  python 03_reconstruction/run_sfm.py <frames_dir> --stage map        # re-run a single stage
"""

import argparse
import json
import time
from pathlib import Path

import pycolmap


def stage_features(database: Path, image_dir: Path, calib: dict | None, device: str):
    t0 = time.perf_counter()
    reader_opts = pycolmap.ImageReaderOptions()
    if calib:
        reader_opts.camera_model = calib["camera_model"]
        reader_opts.camera_params = ",".join(str(p) for p in calib["camera_params"])
    else:
        reader_opts.camera_model = "SIMPLE_RADIAL"
    pycolmap.extract_features(
        database_path=str(database),
        image_path=str(image_dir),
        camera_mode=pycolmap.CameraMode.SINGLE,   # one shared camera (single video)
        reader_options=reader_opts,
        device=getattr(pycolmap.Device, device),
    )
    print(f"[features] done in {time.perf_counter() - t0:.0f}s "
          f"(camera_params={'fixed from calibration' if calib else 'self-calibrated'})")


def stage_match(database: Path, overlap: int, device: str):
    t0 = time.perf_counter()
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = overlap
    pycolmap.match_sequential(database_path=str(database), pairing_options=pairing,
                              device=getattr(pycolmap.Device, device))
    print(f"[match] done in {time.perf_counter() - t0:.0f}s")


def stage_map(database: Path, image_dir: Path, sparse_dir: Path,
             calib: dict | None, use_gpu: bool):
    t0 = time.perf_counter()
    sparse_dir.mkdir(parents=True, exist_ok=True)
    options = pycolmap.IncrementalPipelineOptions()
    if calib:
        # keep the marker-measured intrinsics fixed instead of letting bundle
        # adjustment re-guess focal length/distortion during mapping
        options.ba_refine_focal_length = False
        options.ba_refine_principal_point = False
        options.ba_refine_extra_params = False
    options.ba_use_gpu = use_gpu
    maps = pycolmap.incremental_mapping(
        database_path=str(database),
        image_path=str(image_dir),
        output_path=str(sparse_dir),
        options=options,
    )
    print(f"[map] done in {time.perf_counter() - t0:.0f}s")
    if not maps:
        print("[map] FAILED: no reconstruction produced")
        return
    for idx, rec in maps.items():
        print(f"  model {idx}: {rec.num_reg_images()} images registered, "
              f"{rec.num_points3D()} 3D points, "
              f"mean track length {rec.compute_mean_track_length():.1f}, "
              f"mean reproj error {rec.compute_mean_reprojection_error():.2f}px")


def main():
    parser = argparse.ArgumentParser(description="SfM via pycolmap")
    parser.add_argument("frames_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output dir (default: output/sfm/<frames_dir name>)")
    parser.add_argument("--stage", default="all",
                        choices=["all", "features", "match", "map"])
    parser.add_argument("--overlap", type=int, default=10,
                        help="Sequential matching overlap window (default 10)")
    parser.add_argument("--calib", type=Path, default=None,
                        help="calibrate_camera.py output JSON — fixes intrinsics "
                             "instead of self-calibrating")
    parser.add_argument("--gpu", action="store_true",
                        help="Use CUDA for feature extraction/matching/bundle adjustment")
    args = parser.parse_args()

    out_dir = args.output or Path("output/sfm") / args.frames_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    database = out_dir / "database.db"
    sparse_dir = out_dir / "sparse"

    calib = json.loads(args.calib.read_text()) if args.calib else None
    device = "cuda" if args.gpu else "cpu"

    n_images = len(list(args.frames_dir.glob("*.jpg")))
    print(f"Frames : {n_images} in {args.frames_dir}")
    print(f"Output : {out_dir}")
    print(f"Stage  : {args.stage}")
    print(f"Device : {device}")
    print(f"Calib  : {args.calib if calib else 'none (self-calibrated intrinsics)'}\n")

    if args.stage in ("all", "features"):
        stage_features(database, args.frames_dir, calib, device)
    if args.stage in ("all", "match"):
        stage_match(database, args.overlap, device)
    if args.stage in ("all", "map"):
        stage_map(database, args.frames_dir, sparse_dir, calib, args.gpu)


if __name__ == "__main__":
    main()
