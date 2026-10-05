"""
run_sfm.py — Structure-from-Motion baseline via pycolmap.

Pipeline: SIFT feature extraction → sequential matching (video ordering)
→ incremental mapping. Produces a sparse COLMAP model that is later scaled
to metric units with the fiducial marker (see detect_marker.py).

Camera intrinsics: by default COLMAP self-calibrates focal length as a free
parameter during bundle adjustment, which is a known source of systematic
scale bias. Pass --calib with a calibrate_camera.py output to fix intrinsics
to a value measured directly from the marker (Zhang's method) instead.

GPU note (--gpu)
-----------------
The `pip install pycolmap` wheel on this platform is CPU-only — check with
`python -c "import pycolmap; print(pycolmap.COLMAP_build)"`, it will print
"...without CUDA". Passing device=cuda to pycolmap.extract_features()/
match_sequential() raises "Cannot use GPU feature extraction without CUDA or
OpenGL support" outright; there is no silent CPU fallback. So --gpu here
does NOT call pycolmap's Python GPU path — it shells out to a separately
built/downloaded CUDA-enabled `colmap` CLI binary for feature extraction,
matching, and bundle adjustment (set COLMAP_BIN, e.g.
C:\\thermovation-repos\\colmap-cuda\\bin\\colmap.exe on this machine), the
same approach 03_reconstruction/run_mvs.py uses for dense reconstruction.
Without --gpu, everything still runs through pycolmap exactly as before —
this only changes behavior when --gpu is explicitly requested.

Usage:
  python 03_reconstruction/run_sfm.py dataset/frames/<video_(sfm)> --output output/sfm/video
  python 03_reconstruction/run_sfm.py <frames_dir> --output <out> --calib output/calib/video.json --gpu
  python 03_reconstruction/run_sfm.py <frames_dir> --stage map        # re-run a single stage
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pycolmap

COLMAP_BIN = os.environ.get("COLMAP_BIN", "colmap")


def _resolve_cuda_colmap() -> str:
    resolved = shutil.which(COLMAP_BIN) or (COLMAP_BIN if Path(COLMAP_BIN).exists() else None)
    if not resolved:
        raise SystemExit(
            f"--gpu requires a CUDA-enabled colmap binary; COLMAP_BIN={COLMAP_BIN!r} not found. "
            f'Set it, e.g. $env:COLMAP_BIN = "C:\\thermovation-repos\\colmap-cuda\\bin\\colmap.exe"'
        )
    out = subprocess.run([resolved, "-h"], capture_output=True, text=True, timeout=15)
    if "with CUDA" not in out.stdout:
        raise SystemExit(
            f"--gpu requires a CUDA-enabled colmap binary; {resolved!r} was built without CUDA "
            f"(pycolmap's own GPU path is unavailable for the same reason — see module docstring)."
        )
    return resolved


def _run_cli(cmd: list, log_prefix: str) -> None:
    print(f"[{log_prefix}] $ {' '.join(str(c) for c in cmd)}")
    res = subprocess.run([str(c) for c in cmd])
    if res.returncode != 0:
        raise RuntimeError(f"{log_prefix} failed (exit {res.returncode})")


def stage_features(database: Path, image_dir: Path, calib: dict | None, device: str):
    t0 = time.perf_counter()
    if device == "cuda":
        colmap_bin = _resolve_cuda_colmap()
        cmd = [
            colmap_bin, "feature_extractor",
            "--database_path", database,
            "--image_path", image_dir,
            "--ImageReader.single_camera", "1",   # matches CameraMode.SINGLE below
            "--FeatureExtraction.use_gpu", "1",
        ]
        if calib:
            cmd += [
                "--ImageReader.camera_model", calib["camera_model"],
                "--ImageReader.camera_params", ",".join(str(p) for p in calib["camera_params"]),
            ]
        else:
            cmd += ["--ImageReader.camera_model", "SIMPLE_RADIAL"]
        _run_cli(cmd, "features(gpu)")
    else:
        reader_opts = pycolmap.ImageReaderOptions()
        if calib:
            reader_opts.camera_model = calib["camera_model"]
            reader_opts.camera_params = ",".join(str(p) for p in calib["camera_params"])
        else:
            reader_opts.camera_model = "SIMPLE_RADIAL"
        pycolmap.extract_features(
            database_path=str(database),
            image_path=str(image_dir),
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader_opts,
            device=pycolmap.Device.cpu,
        )
    print(f"[features] done in {time.perf_counter() - t0:.0f}s "
          f"(camera_params={'fixed from calibration' if calib else 'self-calibrated'})")


def stage_match(database: Path, overlap: int, device: str):
    t0 = time.perf_counter()
    if device == "cuda":
        colmap_bin = _resolve_cuda_colmap()
        cmd = [
            colmap_bin, "sequential_matcher",
            "--database_path", database,
            "--SequentialMatching.overlap", str(overlap),
            "--FeatureMatching.use_gpu", "1",
        ]
        _run_cli(cmd, "match(gpu)")
    else:
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = overlap
        pycolmap.match_sequential(database_path=str(database), pairing_options=pairing,
                                  device=pycolmap.Device.cpu)
    print(f"[match] done in {time.perf_counter() - t0:.0f}s")


def _model_stats(rec: pycolmap.Reconstruction, n_total_images: int) -> dict:
    n_reg = rec.num_reg_images()
    return dict(
        n_registered=n_reg,
        n_total_images=n_total_images,
        registration_rate=round(n_reg / n_total_images, 4) if n_total_images else None,
        n_points3d=rec.num_points3D(),
        mean_track_length=round(float(rec.compute_mean_track_length()), 2),
        mean_reprojection_error_px=round(float(rec.compute_mean_reprojection_error()), 3),
    )


def stage_map(database: Path, image_dir: Path, sparse_dir: Path,
             calib: dict | None, use_gpu: bool):
    t0 = time.perf_counter()
    sparse_dir.mkdir(parents=True, exist_ok=True)
    n_total_images = len(list(image_dir.glob("*.jpg")))
    stats_by_model: dict[str, dict] = {}
    if use_gpu:
        colmap_bin = _resolve_cuda_colmap()
        cmd = [
            colmap_bin, "mapper",
            "--database_path", database,
            "--image_path", image_dir,
            "--output_path", sparse_dir,
            "--Mapper.ba_use_gpu", "1",
        ]
        if calib:
            cmd += [
                "--Mapper.ba_refine_focal_length", "0",
                "--Mapper.ba_refine_principal_point", "0",
                "--Mapper.ba_refine_extra_params", "0",
            ]
        _run_cli(cmd, "map(gpu)")
        print(f"[map] done in {time.perf_counter() - t0:.0f}s")
        model_dirs = sorted(p for p in sparse_dir.iterdir()
                            if p.is_dir() and (p / "images.bin").exists())
        if not model_dirs:
            print("[map] FAILED: no reconstruction produced")
            return
        for model_dir in model_dirs:
            rec = pycolmap.Reconstruction(str(model_dir))
            stats = _model_stats(rec, n_total_images)
            stats_by_model[model_dir.name] = stats
            print(f"  model {model_dir.name}: {stats['n_registered']} images registered, "
                  f"{stats['n_points3d']} 3D points, "
                  f"mean track length {stats['mean_track_length']:.1f}, "
                  f"mean reproj error {stats['mean_reprojection_error_px']:.2f}px")
    else:
        options = pycolmap.IncrementalPipelineOptions()
        if calib:
            # keep the marker-measured intrinsics fixed instead of letting bundle
            # adjustment re-guess focal length/distortion during mapping
            options.ba_refine_focal_length = False
            options.ba_refine_principal_point = False
            options.ba_refine_extra_params = False
        options.ba_use_gpu = False
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
            stats = _model_stats(rec, n_total_images)
            stats_by_model[str(idx)] = stats
            print(f"  model {idx}: {stats['n_registered']} images registered, "
                  f"{stats['n_points3d']} 3D points, "
                  f"mean track length {stats['mean_track_length']:.1f}, "
                  f"mean reproj error {stats['mean_reprojection_error_px']:.2f}px")

    # Best model = most registered images, matching run.py's best_sparse_model().
    if stats_by_model:
        best_name = max(stats_by_model, key=lambda k: stats_by_model[k]["n_registered"])
        out = dict(models=stats_by_model, best_model=best_name)
        (database.parent / "sfm_stats.json").write_text(json.dumps(out, indent=2))


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
                        help="Use CUDA for feature extraction/matching/bundle adjustment "
                             "(via a separate CUDA colmap.exe — see COLMAP_BIN; the PyPI "
                             "pycolmap wheel is CPU-only, see module docstring)")
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
