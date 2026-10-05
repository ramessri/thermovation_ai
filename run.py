"""
run.py — end-to-end Thermovation assessment pipeline.

  video → frames → SfM → marker metric scale → room dimensions
  → room-frame alignment (--align) → [wip] placement recommendation (--placement)

Each stage is cached: if its output already exists it is skipped (use --force
to redo everything). Multiple videos can be processed in one call and run in
parallel worker processes (--parallel N).

DATASET_DIR and OUTPUT_DIR can be set in a .env file (see .env.example) so
paths don't need to be retyped on every call. Video args without a path
separator (e.g. "IMG_3126.mov") resolve against DATASET_DIR; anything with
a slash/backslash or an absolute path is used as-is.

Usage:
  python run.py IMG_3126.mov
  python run.py dataset/IMG_3126.mov
  python run.py "*.mov" --parallel 2
  python run.py video.mp4 --fps 2 --max-dim 1440 --force
  python run.py "*.mov" --calibrate --gpu --output output/pipeline_calibrated
  python run.py IMG_3126.mov --align       # also build the marker-anchored room frame + top-down
  python run.py IMG_3126.mov --placement   # also run placement recommendation
  python run.py IMG_3126.mov --pipes --placement  # pipes runs first, feeds Ruecklauf pairing into placement
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Progress/status lines below print real German/accented text ("Rücklauf",
# °, ×) — Windows consoles and pipes default to a legacy codepage (cp1252)
# that can't encode them, crashing the whole run on the first such
# character. Force UTF-8, same fix as the sibling photogram project's
# scripts/smoke.py. Also set PYTHONIOENCODING so every stage subprocess
# this script spawns (run_sfm.py, placement_3d.py, etc.) inherits it too —
# their own stdout needs the same fix, not just this top-level process's.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PY = sys.executable
ROOT = Path(__file__).parent
DEFAULT_DATASET_DIR = Path(os.environ.get("DATASET_DIR", "dataset"))
DEFAULT_OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "output/pipeline"))


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
                  force: bool, calibrate: bool, gpu: bool, align: bool,
                  pipes: bool, placement: bool, dense: bool = False,
                  fix_jumps: bool = False, marker_type: str = "custom",
                  export: bool = False, coverage: bool = False) -> dict:
    stem = video.stem.replace(" ", "_")
    work = out_root / stem
    status = dict(video=video.name, work_dir=str(work))
    try:
        return _process_video_inner(
            video, out_root, fps, max_dim, force, calibrate, gpu, align,
            pipes, placement, dense, fix_jumps, marker_type, export, coverage, stem, work, status,
        )
    except RuntimeError as e:
        # A required stage (frames/SfM) raised via sh() — previously this
        # propagated all the way out of run.py with no report.json ever
        # written, so a caller reading report.json from disk (the FastAPI
        # wrapper) saw a clean process exit with nothing to read and had to
        # guess. Always leave a report behind, and print an explicit
        # "FAILED: ..." line the wrapper's log-tail fallback can also catch.
        status["error"] = str(e)
        status["runtime_s"] = round(time.perf_counter() - status.get("_t_start", time.perf_counter()), 1)
        status.pop("_t_start", None)
        print(f"FAILED: {e}")
        report = work / "report.json"
        work.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(status, indent=2))
        status["report"] = str(report)
        return status


def _process_video_inner(video: Path, out_root: Path, fps: float, max_dim: int,
                         force: bool, calibrate: bool, gpu: bool, align: bool,
                         pipes: bool, placement: bool, dense: bool, fix_jumps: bool,
                         marker_type: str, export: bool, coverage: bool,
                         stem: str, work: Path, status: dict) -> dict:
    frames_dir = work / "frames"
    sfm_dir = work / "sfm"
    calib_path = work / "calib.json"
    t_start = time.perf_counter()
    status["_t_start"] = t_start

    def fail(msg: str) -> dict:
        status["error"] = msg
        status["runtime_s"] = round(time.perf_counter() - t_start, 1)
        status.pop("_t_start", None)
        print(f"FAILED: {msg}")
        report = work / "report.json"
        work.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(status, indent=2))
        status["report"] = str(report)
        return status

    # 1. frame extraction
    if force or not any(frames_dir.glob("*.jpg")):
        sh([PY, "01_frames/extract_frames.py", video, "--out", frames_dir,
            "--fps", fps, "--max-dim", max_dim], f"{stem}:frames")
    n_frames = len(list(frames_dir.glob("*.jpg")))
    status["frames"] = n_frames

    # 2. camera calibration from marker sightings (custom-board only — Zhang's
    # method needs the board's known multi-square geometry across many views;
    # ArUco markers are typically smaller/scattered and aren't a good target
    # for this specific calibration approach, so ArUco mode self-calibrates
    # via bundle adjustment instead, same as photogram does by default)
    if calibrate and marker_type == "custom":
        if force or not calib_path.exists():
            print(f"[{stem}:calibrate]")
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
    elif calibrate and marker_type == "aruco":
        status["calibration_skipped"] = True
        status["calibration_skip_reason"] = "marker_type=aruco self-calibrates (no Zhang target)"

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
        return fail("SfM produced no reconstruction")
    status["sparse_model"] = model_dir.name

    # 3b. dense reconstruction (opt-in via --dense; best-effort — CUDA-only,
    # see 03_reconstruction/run_mvs.py; failure never blocks the rest of the
    # pipeline since scale/room_dims/placement all still work off the sparse model)
    dense_ply = None
    if dense:
        dense_ply = sfm_dir / "dense" / "fused.ply"
        if force or not dense_ply.exists():
            print(f"[{stem}:dense]")
            res = subprocess.run(
                [str(c) for c in [PY, "03_reconstruction/run_mvs.py", sfm_dir, frames_dir,
                                  "--model", model_dir.name]],
                cwd=ROOT,
            )
            if res.returncode != 0:
                print(f"[{stem}:dense] errored (non-fatal, sparse-only downstream)")
        if dense_ply.exists():
            status["dense_cloud"] = str(dense_ply)
        else:
            status["dense_skipped"] = True

    # 3c. trajectory-jump detection + correction (opt-in via --fix-jumps;
    # best-effort — a mis-registered "teleport" block from a tracking break
    # can throw off both marker scale (if the marker was seen in an affected
    # frame) and room_dims/placement geometry. Writes a corrected sparse
    # model and, if present, a corrected dense cloud; downstream stages then
    # use the corrected model transparently via model_dir.
    if fix_jumps:
        fixed_model_dir = sfm_dir / "sparse" / f"{model_dir.name}_fixed"
        jumps_json = sfm_dir / "trajectory_jumps.json"
        if force or not jumps_json.exists():
            print(f"[{stem}:fix_jumps]")
            res = subprocess.run(
                [str(c) for c in [PY, "03_reconstruction/fix_trajectory_jumps.py", sfm_dir, frames_dir,
                                  "--model", model_dir.name]],
                cwd=ROOT,
            )
            if res.returncode != 0:
                print(f"[{stem}:fix_jumps] errored (non-fatal, using uncorrected model)")
        if jumps_json.exists():
            status["trajectory_jumps"] = json.loads(jumps_json.read_text())
            if fixed_model_dir.exists():
                model_dir = fixed_model_dir
                status["sparse_model"] = model_dir.name
                status["trajectory_jumps_corrected"] = True
        else:
            status["fix_jumps_skipped"] = True

    # A jump correction rewrote model_dir mid-run — cached outputs from the
    # ORIGINAL (uncorrected) model are now stale even if force wasn't passed.
    redo = force or status.get("trajectory_jumps_corrected", False)

    # 4. metric scale from marker
    scale_script = "04_scale/scale_sfm.py" if marker_type == "custom" else "04_scale/scale_sfm_aruco.py"
    if redo or not (sfm_dir / "scale.json").exists():
        sh([PY, scale_script, model_dir, frames_dir], f"{stem}:scale")
    if not (sfm_dir / "scale.json").exists():
        return fail(f"no reliable {marker_type} marker scale")
    status["scale"] = json.loads((sfm_dir / "scale.json").read_text())

    # 5. room dimensions
    if redo or not (sfm_dir / "room_dims.json").exists():
        dims_cmd = [PY, "05_geometry/room_dims.py", sfm_dir, "--model", model_dir.name]
        if dense_ply and dense_ply.exists():
            dims_cmd += ["--dense-ply", dense_ply]
        sh(dims_cmd, f"{stem}:dims")
    if (sfm_dir / "room_dims.json").exists():
        status["room_dims"] = json.loads((sfm_dir / "room_dims.json").read_text())

    # 5b. scan-completeness scoring + re-shoot suggestions (opt-in via
    # --coverage; best-effort — HPR occlusion scoring + DBSCAN clustering,
    # see 10_quality/coverage.py)
    if coverage:
        coverage_json = sfm_dir / "coverage.json"
        if redo or not coverage_json.exists():
            print(f"[{stem}:coverage]")
            coverage_cmd = [PY, "10_quality/coverage.py", sfm_dir, "--model", model_dir.name]
            if dense_ply and dense_ply.exists():
                coverage_cmd += ["--dense-ply", dense_ply]
            res = subprocess.run([str(c) for c in coverage_cmd], cwd=ROOT)
            if res.returncode != 0:
                print(f"[{stem}:coverage] errored (non-fatal)")
        if coverage_json.exists():
            status["coverage"] = json.loads(coverage_json.read_text())
        else:
            status["coverage_skipped"] = True

    # 6. room-frame alignment (opt-in via --align; best-effort — needs marker_triangulation
    #    scale, so it's a no-op for videos that fell back to depth_ratio scaling)
    if align:
        topdown_path = sfm_dir / "aligned_room" / "topdown.png"
        if redo or not topdown_path.exists():
            print(f"[{stem}:align]")
            for script in ["05_alignment/align_world.py", "05_alignment/align_room.py"]:
                res = subprocess.run(
                    [str(c) for c in [PY, script, sfm_dir]], cwd=ROOT)
                if res.returncode != 0:
                    print(f"[{stem}:align] {script} errored (non-fatal)")
        if topdown_path.exists():
            status["aligned_room_topdown"] = str(topdown_path)
        else:
            status["align_skipped"] = True

    # pipe paths & lengths (opt-in via --pipes; best-effort — runs BEFORE
    # placement so placement_3d.py can read pipe_lengths.json's Vor-/
    # Ruecklauf pairing signal when both flags are passed together)
    if pipes:
        pipe_lengths_json = sfm_dir / "pipe_lengths.json"
        if redo or not pipe_lengths_json.exists():
            print(f"[{stem}:pipes]")
            res = subprocess.run(
                [str(c) for c in [PY, "07_pipes/pipe_paths.py", sfm_dir, frames_dir,
                                  "--model", model_dir.name]],
                cwd=ROOT,
            )
            if res.returncode != 0:
                print(f"[{stem}:pipes] errored (non-fatal)")
        if pipe_lengths_json.exists():
            status["pipes"] = json.loads(pipe_lengths_json.read_text())
        else:
            status["pipes_skipped"] = True

    # 7. placement recommendation (opt-in via --placement; best-effort — skip if no clean wall)
    if placement:
        placement_json = sfm_dir / "placement_3d.json"
        if redo or not placement_json.exists():
            print(f"[{stem}:placement]")
            placement_cmd = [PY, "06_placement/placement_3d.py", sfm_dir, frames_dir,
                             "--model", model_dir.name]
            if dense_ply and dense_ply.exists():
                placement_cmd += ["--dense-ply", dense_ply]
            res = subprocess.run(
                [str(c) for c in placement_cmd],
                cwd=ROOT,
            )
            if res.returncode != 0:
                print(f"[{stem}:placement] errored (non-fatal)")
        if placement_json.exists():
            status["placement"] = json.loads(placement_json.read_text())
        else:
            status["placement_skipped"] = True

    # 8. mesh + point cloud export (opt-in via --export; best-effort — needs
    # a real deliverable file, not just JSON numbers, to open in Blender/
    # MeshLab/CAD; see 09_export/export_mesh.py for why BPA not Poisson)
    if export:
        export_json = sfm_dir / "export" / "export.json"
        if redo or not export_json.exists():
            print(f"[{stem}:export]")
            export_cmd = [PY, "09_export/export_mesh.py", sfm_dir, "--model", model_dir.name]
            if dense_ply and dense_ply.exists():
                export_cmd += ["--dense-ply", dense_ply]
            res = subprocess.run([str(c) for c in export_cmd], cwd=ROOT)
            if res.returncode != 0:
                print(f"[{stem}:export] errored (non-fatal)")
        if export_json.exists():
            status["export"] = json.loads(export_json.read_text())
        else:
            status["export_skipped"] = True

    # 9. composite quality score — always runs, pure JSON aggregation over
    # whatever the stages above produced (see 10_quality/quality_score.py)
    print(f"[{stem}:quality]")
    quality_res = subprocess.run(
        [str(c) for c in [PY, "10_quality/quality_score.py", sfm_dir, "--model", model_dir.name]],
        cwd=ROOT,
    )
    quality_json = sfm_dir / "quality_score.json"
    if quality_res.returncode == 0 and quality_json.exists():
        status["quality_score"] = json.loads(quality_json.read_text())

    status["runtime_s"] = round(time.perf_counter() - t_start, 1)
    status.pop("_t_start", None)

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
        if r.get("aligned_room_topdown"):
            print(f"  align: room frame + top-down saved to {r['aligned_room_topdown']}")
        elif r.get("align_skipped"):
            print("  align: skipped (needs marker_triangulation scale, not depth_ratio_fallback)")
        pipes = r.get("pipes")
        if pipes and pipes.get("pipes"):
            n_pipes = len(pipes["pipes"])
            total_len = sum(p["length_cm"] for p in pipes["pipes"])
            pair = pipes.get("rucklauf_pipe_pairing", {})
            pair_note = (f", Vor-/Ruecklauf pair found (score={pair['score']})"
                        if pair.get("pair_found") else "")
            print(f"  pipes: {n_pipes} instance(s), {total_len:.0f}cm total{pair_note}")
        elif r.get("pipes_skipped"):
            print("  pipes: skipped (no pipe masks detected)")
        placement = r.get("placement")
        if placement and placement.get("top3"):
            best = placement["top3"][0]
            print(f"  place: best unit spot d(Rücklauf)={best['dist_to_rucklauf_cm']:.0f}cm "
                  f"clearance={best['clearance_cm']:.0f}cm height={best['mount_height_cm']:.0f}cm")
        elif r.get("placement_skipped"):
            print("  place: skipped (no clean single-wall marker for a reliable plane)")
        if r.get("dense_cloud"):
            print(f"  dense: {r['dense_cloud']}")
        elif r.get("dense_skipped"):
            print("  dense: skipped (needs CUDA colmap.exe — see COLMAP_BIN)")
        jumps = r.get("trajectory_jumps")
        if jumps:
            n = len(jumps.get("jumps_applied", []))
            if n:
                print(f"  jumps: corrected {n} trajectory jump(s) — sparse model replaced")
            else:
                print(f"  jumps: {len(jumps.get('jumps_skipped', []))} candidate(s) checked, none corrected")
        export = r.get("export")
        if export and export.get("obj"):
            print(f"  export: {export['n_mesh_vertices']:,} verts / {export['n_mesh_triangles']:,} tris "
                  f"({export['source']} cloud) -> {export['obj']}")
        elif r.get("export_skipped"):
            print("  export: skipped (too few points, or Ball Pivoting produced no triangles)")
        cov = r.get("coverage")
        if cov:
            print(f"  coverage: {cov['coverage_score']*100:.0f}% — {cov['n_suggestions']} re-shoot suggestion(s)")
        elif r.get("coverage_skipped"):
            print("  coverage: skipped")
        q = r.get("quality_score")
        if q and q.get("quality_score") is not None:
            print(f"  quality: {q['quality_score']:.0f}/100")


def main():
    parser = argparse.ArgumentParser(description="End-to-end assessment pipeline")
    parser.add_argument("videos", nargs="+",
                        help="Video file(s) or glob pattern(s). A bare name with no "
                             "path separator resolves against --dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR,
                        help=f"Folder bare video names resolve against (default: "
                             f"{DEFAULT_DATASET_DIR}, or DATASET_DIR in .env)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"default: {DEFAULT_OUTPUT_DIR}, or OUTPUT_DIR in .env")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-dim", type=int, default=1440)
    parser.add_argument("--parallel", type=int, default=1,
                        help="Process N videos concurrently (default 1)")
    parser.add_argument("--force", action="store_true", help="Redo cached stages")
    parser.add_argument("--calibrate", action="store_true",
                        help="Calibrate camera intrinsics from marker sightings "
                             "(fixes focal length instead of self-calibrating). "
                             "Custom-board only — no-op (self-calibrates) with --marker-type aruco.")
    parser.add_argument("--marker-type", default="custom", choices=["custom", "aruco"],
                        help="'custom' (default): the printed 3x3 boiler-room board, scale from "
                             "04_scale/scale_sfm.py. 'aruco': OpenCV ArUco markers (DICT_4X4_100 "
                             "by default, ARUCO_DICT/ARUCO_MARKER_SIZE_CM env vars), scale from "
                             "04_scale/scale_sfm_aruco.py — needs 2+ markers with at least two "
                             "co-visible in one frame (same algorithm as the sibling photogram "
                             "project's ArUco pipeline: physical baseline vs SfM-unit triangulated "
                             "distance between marker pairs, not single-marker known-size geometry).")
    parser.add_argument("--gpu", action="store_true",
                        help="Use CUDA for SfM feature extraction/matching/mapping")
    parser.add_argument("--align", action="store_true",
                        help="Also build the marker-anchored, gravity-up room frame + "
                             "top-down floor plan (needs marker_triangulation scale)")
    parser.add_argument("--pipes", action="store_true",
                        help="Also run pipe path/length extraction + Vor-/Ruecklauf "
                             "color pairing (runs before --placement if both are set)")
    parser.add_argument("--placement", action="store_true",
                        help="Also run the placement recommendation stage "
                             "(default: pipeline stops after room dimensions)")
    parser.add_argument("--dense", action="store_true",
                        help="Also run CUDA dense reconstruction (patch-match stereo + "
                             "fusion) after SfM, producing sfm_dir/dense/fused.ply. "
                             "Best-effort/non-fatal; requires COLMAP_BIN (a CUDA-enabled "
                             "colmap.exe — the PyPI pycolmap wheel is CPU-only, see "
                             "03_reconstruction/run_mvs.py). When it succeeds, room_dims.py "
                             "uses the dense cloud for footprint/height (more points on "
                             "textureless walls/floors than sparse SIFT features give) and "
                             "placement_3d.py uses it for wall-surface evidence instead of "
                             "single-frame monocular depth (--dense-ply on both scripts).")
    parser.add_argument("--fix-jumps", action="store_true",
                        help="Detect and correct mis-registered 'teleport' blocks caused by "
                             "a tracking break (see 03_reconstruction/fix_trajectory_jumps.py). "
                             "Best-effort/non-fatal; writes a corrected sparse model that "
                             "scale/room_dims/align/pipes/placement then use instead.")
    parser.add_argument("--export", action="store_true",
                        help="Export a real deliverable: colored point cloud (PLY/LAS) + a "
                             "Ball-Pivoting mesh (OBJ) to sfm_dir/export/ (see "
                             "09_export/export_mesh.py). Best-effort/non-fatal; uses --dense's "
                             "cloud when available for a much better mesh.")
    parser.add_argument("--coverage", action="store_true",
                        help="Score scan completeness (HPR occlusion + view-angle scoring) and "
                             "generate re-shoot suggestions for under-covered areas, colored "
                             "cloud + coverage.json (see 10_quality/coverage.py). "
                             "Best-effort/non-fatal; uses --dense's cloud when available.")
    args = parser.parse_args()

    # Bare names (no path separator) resolve against --dataset; anything
    # with a slash/backslash or an absolute path is used as typed.
    resolved = [v if (os.path.isabs(v) or "/" in v or "\\" in v) else str(args.dataset / v)
                for v in args.videos]

    # cmd.exe/PowerShell don't expand wildcards like bash does, so expand
    # any unexpanded glob patterns (e.g. "dataset/*.mov") ourselves.
    videos = []
    for v in resolved:
        matches = sorted(Path(p) for p in glob.glob(v))
        videos.extend(matches if matches else [Path(v)])
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
            if args.align:
                cmd.append("--align")
            if args.pipes:
                cmd.append("--pipes")
            if args.placement:
                cmd.append("--placement")
            if args.dense:
                cmd.append("--dense")
            if args.fix_jumps:
                cmd.append("--fix-jumps")
            if args.export:
                cmd.append("--export")
            if args.coverage:
                cmd.append("--coverage")
            cmd += ["--marker-type", args.marker_type]
            procs.append((v, subprocess.Popen(cmd, cwd=ROOT)))
            while sum(p.poll() is None for _, p in procs) >= args.parallel:
                time.sleep(2)
        for _, p in procs:
            p.wait()
        return

    results = [process_video(v, args.output, args.fps, args.max_dim, args.force,
                             args.calibrate, args.gpu, args.align, args.pipes, args.placement,
                             args.dense, args.fix_jumps, args.marker_type, args.export, args.coverage)
               for v in args.videos]
    print_summary(results)


if __name__ == "__main__":
    main()
