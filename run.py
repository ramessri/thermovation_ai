"""
run.py — end-to-end Thermovation assessment pipeline.

  video → frames → SfM → marker metric scale → room dimensions
  → [wip] placement recommendation (--placement)

Each stage is cached: if its output already exists it is skipped (use --force
to redo everything). Multiple videos can be processed in one call and run in
parallel worker processes (--parallel N).

Usage:
  python run.py dataset/IMG_3126.mov
  python run.py dataset/*.mov --parallel 2
  python run.py dataset/video.mp4 --fps 2 --max-dim 1440 --force
  python run.py dataset/*.mov --calibrate --gpu --output output/pipeline_calibrated
  python run.py dataset/IMG_3126.mov --placement   # also run placement recommendation
"""

import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).parent


def sh(cmd: list, log_prefix: str) -> None:
    cmd = [str(c) for c in cmd]
    print(f"[{log_prefix}] $ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=ROOT)
    if res.returncode != 0:
        raise RuntimeError(f"stage '{log_prefix}' failed (exit {res.returncode})")


def best_sparse_model(sfm_dir: Path) -> Path | None:
    """Pick the sub-model with the most registered images."""
    import pycolmap
    best, best_n = None, -1
    sparse = sfm_dir / "sparse"
    if not sparse.exists():
        return None
    for model_dir in sorted(sparse.iterdir()):
        if not (model_dir / "images.bin").exists():
            continue
        rec = pycolmap.Reconstruction(str(model_dir))
        if rec.num_reg_images() > best_n:
            best, best_n = model_dir, rec.num_reg_images()
    return best


def process_video(video: Path, out_root: Path, fps: float, max_dim: int,
                  force: bool, calibrate: bool, gpu: bool, placement: bool) -> dict:
    stem = video.stem.replace(" ", "_")
    work = out_root / stem
    frames_dir = work / "frames"
    sfm_dir = work / "sfm"
    calib_path = work / "calib.json"
    t_start = time.perf_counter()
    status = dict(video=video.name, work_dir=str(work))

    # 1. frame extraction
    if force or not any(frames_dir.glob("*.jpg")):
        sh([PY, "01_frames/extract_frames.py", video, "--out", frames_dir,
            "--fps", fps, "--max-dim", max_dim], f"{stem}:frames")
    n_frames = len(list(frames_dir.glob("*.jpg")))
    status["frames"] = n_frames

    # 2. camera calibration from marker sightings
    if calibrate:
        if force or not calib_path.exists():
            res = subprocess.run(
                [str(c) for c in [PY, "02_calibration/calibrate_camera.py", frames_dir,
                                  "--output", calib_path]],
                cwd=ROOT,
            )
            if res.returncode != 0:
                print(f"[{stem}:calibrate] errored (non-fatal, will self-calibrate)")
        if calib_path.exists():
            status["calibration"] = json.loads(calib_path.read_text())
        else:
            status["calibration_skipped"] = True

    # 3. SfM
    if force or not (sfm_dir / "sparse").exists():
        cmd = [PY, "03_reconstruction/run_sfm.py", frames_dir, "--output", sfm_dir]
        if calibrate and calib_path.exists():
            cmd += ["--calib", calib_path]
        if gpu:
            cmd += ["--gpu"]
        sh(cmd, f"{stem}:sfm")
    model_dir = best_sparse_model(sfm_dir)
    if model_dir is None:
        status["error"] = "SfM produced no reconstruction"
        return status
    status["sparse_model"] = model_dir.name

    # 4. metric scale from marker
    if force or not (sfm_dir / "scale.json").exists():
        sh([PY, "04_scale/scale_sfm.py", model_dir, frames_dir], f"{stem}:scale")
    if not (sfm_dir / "scale.json").exists():
        status["error"] = "no reliable marker scale"
        return status
    status["scale"] = json.loads((sfm_dir / "scale.json").read_text())

    # 5. room dimensions
    if force or not (sfm_dir / "room_dims.json").exists():
        sh([PY, "05_geometry/room_dims.py", sfm_dir, "--model", model_dir.name],
           f"{stem}:dims")
    if (sfm_dir / "room_dims.json").exists():
        status["room_dims"] = json.loads((sfm_dir / "room_dims.json").read_text())

    # 6. placement recommendation (opt-in via --placement; best-effort — skip if no clean wall)
    if placement:
        placement_json = sfm_dir / "placement_3d.json"
        if force or not placement_json.exists():
            res = subprocess.run(
                [str(c) for c in [PY, "06_placement/placement_3d.py", sfm_dir, frames_dir,
                                  "--model", model_dir.name]],
                cwd=ROOT,
            )
            if res.returncode != 0:
                print(f"[{stem}:placement] errored (non-fatal)")
        if placement_json.exists():
            status["placement"] = json.loads(placement_json.read_text())
        else:
            status["placement_skipped"] = True

    status["runtime_s"] = round(time.perf_counter() - t_start, 1)

    # 7. consolidated report
    report = work / "report.json"
    report.write_text(json.dumps(status, indent=2))
    status["report"] = str(report)
    return status


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("PIPELINE SUMMARY")
    print("=" * 72)
    for r in results:
        print(f"\n{r['video']}")
        if "error" in r:
            print(f"  FAILED: {r['error']}")
            continue
        scale = r.get("scale", {})
        dims = r.get("room_dims", {})
        calib = r.get("calibration")
        print(f"  frames={r.get('frames')}  model={r.get('sparse_model')}  "
              f"runtime={r.get('runtime_s', '?')}s")
        if calib:
            print(f"  calib: focal={calib['camera_params'][0]:.0f}px "
                  f"RMS={calib['rms_reprojection_error_px']:.2f}px "
                  f"({calib['n_views_used']} views)")
        elif r.get("calibration_skipped"):
            print("  calib: skipped (too few marker sightings) — SfM self-calibrated intrinsics")
        if scale:
            method = scale.get("method", "marker_triangulation")
            if method == "depth_ratio_fallback":
                detail = f"depth-ratio fallback, {scale.get('depth_ratio_frames', '?')} frames"
            else:
                detail = f"{scale.get('segments_accepted', '?')} marker segment(s)"
            print(f"  scale: {scale['cm_per_unit']:.2f} cm/unit ({detail})")
        if dims:
            bound = ">=" if dims.get("height_is_lower_bound") else ""
            print(f"  room : L={dims['length_cm']:.0f}cm  W={dims['width_cm']:.0f}cm  "
                  f"H={bound}{dims['height_cm']:.0f}cm")
        placement = r.get("placement")
        if placement and placement.get("top3"):
            best = placement["top3"][0]
            print(f"  place: best unit spot d(Rücklauf)={best['dist_to_rucklauf_cm']:.0f}cm "
                  f"clearance={best['clearance_cm']:.0f}cm height={best['mount_height_cm']:.0f}cm")
        elif r.get("placement_skipped"):
            print("  place: skipped (no clean single-wall marker for a reliable plane)")


def main():
    parser = argparse.ArgumentParser(description="End-to-end assessment pipeline")
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/pipeline"))
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-dim", type=int, default=1440)
    parser.add_argument("--parallel", type=int, default=1,
                        help="Process N videos concurrently (default 1)")
    parser.add_argument("--force", action="store_true", help="Redo cached stages")
    parser.add_argument("--calibrate", action="store_true",
                        help="Calibrate camera intrinsics from marker sightings "
                             "(fixes focal length instead of self-calibrating)")
    parser.add_argument("--gpu", action="store_true",
                        help="Use CUDA for SfM feature extraction/matching/mapping")
    parser.add_argument("--placement", action="store_true",
                        help="Also run the placement recommendation stage "
                             "(default: pipeline stops after room dimensions)")
    args = parser.parse_args()

    # cmd.exe/PowerShell don't expand wildcards like bash does, so expand
    # any unexpanded glob patterns (e.g. "dataset/*.mov") ourselves.
    videos = []
    for v in args.videos:
        matches = sorted(Path(p) for p in glob.glob(str(v)))
        videos.extend(matches if matches else [v])
    args.videos = videos

    if args.parallel > 1 and len(args.videos) > 1:
        # one worker process per video
        procs = []
        for v in args.videos:
            cmd = [PY, __file__, str(v), "--output", str(args.output),
                   "--fps", str(args.fps), "--max-dim", str(args.max_dim)]
            if args.force:
                cmd.append("--force")
            if args.calibrate:
                cmd.append("--calibrate")
            if args.gpu:
                cmd.append("--gpu")
            if args.placement:
                cmd.append("--placement")
            procs.append((v, subprocess.Popen(cmd, cwd=ROOT)))
            while sum(p.poll() is None for _, p in procs) >= args.parallel:
                time.sleep(2)
        for _, p in procs:
            p.wait()
        return

    results = [process_video(v, args.output, args.fps, args.max_dim, args.force,
                             args.calibrate, args.gpu, args.placement)
               for v in args.videos]
    print_summary(results)


if __name__ == "__main__":
    main()
