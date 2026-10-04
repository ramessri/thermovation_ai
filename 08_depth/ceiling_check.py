"""
ceiling_check.py — does dense monocular depth recover a ceiling estimate
where sparse SfM structurally cannot (9.2b)?

Runs a chosen depth model on a sampled subset of registered frames,
back-projects EVERY pixel of each depth map into 3D using that frame's
already-known camera pose (from the SfM model this video already has —
this validation assumes the existing marker-based cm_per_unit scale is
trustworthy; it is not itself a scale check, see depth_scale.py for that),
and feeds the resulting dense point cloud through the SAME
compute_room_dims() used by the SfM, 3DGS, and NeRF arms — so "does this
recover the ceiling" is decided by the identical gap-based ceiling logic
everywhere else, not a hand-picked heuristic here.

Usage (single video):
  python 08_depth/ceiling_check.py \
      --sfm-dir output/sfm/IMG_3126 --frames-dir dataset/frames/IMG_3126_sfm \
      --model 1 --depth-model depthanything_v2_metric \
      --out 08_depth/results/IMG_3126

Usage (batch — every video under a root with a completed sfm/ + frames/):
  python 08_depth/ceiling_check.py \
      --sfm-root "C:\\thermovation-output\\sfm 1" \
      --out 08_depth/results/ceiling_check_all
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pycolmap

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "05_geometry"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from room_dims import up_from_cameras, gravity_rotation_from_prior, compute_room_dims, render_topdown
from depth_models import DEPTH_MODELS


def best_sparse_model(sfm_dir: Path) -> str | None:
    """Pick the sub-model with the most registered images (same logic as
    run.py's best_sparse_model, duplicated here so batch mode doesn't need
    a hardcoded video->model-index table)."""
    best, best_n = None, -1
    sparse = sfm_dir / "sparse"
    if not sparse.exists():
        return None
    for model_dir in sorted(sparse.iterdir()):
        if not (model_dir / "images.bin").exists():
            continue
        rec = pycolmap.Reconstruction(str(model_dir))
        if rec.num_reg_images() > best_n:
            best, best_n = model_dir.name, rec.num_reg_images()
    return best


def backproject_depth_map(depth_cm: np.ndarray, cam: pycolmap.Camera,
                          cam_from_world: np.ndarray, stride: int = 4) -> np.ndarray:
    """Every `stride`-th pixel of a metric depth map -> world-frame 3D points
    (model's own world frame, in CM — caller already has cm_per_unit-scaled
    camera poses, see process_video()).

    Uses cam.cam_from_img() — the same pycolmap-native pixel-to-ray method
    scale_sfm.py already uses — rather than hand-deriving fx/fy/cx/cy from
    cam.params, since this project's calibrated cameras use COLMAP's OPENCV
    model with real lens distortion (see calibrate_camera.py); a naive
    pinhole-only back-projection would be quietly wrong for those, exactly
    the class of bug the camera-calibration work earlier in this project
    was about."""
    h, w = depth_cm.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    d = depth_cm[ys, xs]
    valid = d > 1e-3
    xs, ys, d = xs[valid].astype(np.float64), ys[valid].astype(np.float64), d[valid]

    pixels = np.stack([xs, ys], axis=1)
    rays_xy = np.asarray(cam.cam_from_img(pixels))   # undistorted normalized (x/z, y/z)
    pts_cam = np.concatenate([rays_xy * d[:, None], d[:, None]], axis=1)   # cm, camera frame

    R_cw = cam_from_world[:, :3]
    t_cw = cam_from_world[:, 3]
    # cam_from_world: X_cam = R_cw @ X_world + t_cw  =>  X_world = R_cw.T @ (X_cam - t_cw)
    pts_world_cm = (pts_cam - t_cw) @ R_cw
    return pts_world_cm


def process_video(sfm_dir: Path, frames_dir: Path, model_name: str, loaded, predict_fn,
                  device: str, depth_model_name: str, every_n_frames: int,
                  pixel_stride: int, out_dir: Path) -> dict:
    scale_path = sfm_dir / "scale.json"
    if not scale_path.exists():
        return dict(error="no scale.json")
    scale_info = json.loads(scale_path.read_text())
    cm_per_unit = scale_info["cm_per_unit"]
    rec = pycolmap.Reconstruction(str(sfm_dir / "sparse" / model_name))
    print(f"  Model: {rec.num_reg_images()} images | scale {cm_per_unit:.3f} cm/unit", flush=True)

    existing_dims_path = sfm_dir / "room_dims.json"
    existing = json.loads(existing_dims_path.read_text()) if existing_dims_path.exists() else None
    if existing:
        bound = ">=" if existing.get("height_is_lower_bound") else ""
        print(f"  Existing SfM-only height: {bound}{existing.get('height_cm')} cm "
              f"(reliable={existing.get('height_reliable')})", flush=True)

    images = sorted(rec.images.values(), key=lambda im: im.name)[::every_n_frames]
    all_pts_cm = []
    frames_used = 0
    for img in images:
        bgr = cv2.imread(str(frames_dir / img.name))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cam = rec.cameras[img.camera_id]
        focal_px = float(cam.params[0])
        depth_cm = predict_fn(rgb, loaded, device, focal_px=focal_px)
        cam_from_world = np.asarray(img.cam_from_world().matrix())
        cam_from_world_cm = cam_from_world.copy()
        cam_from_world_cm[:, 3] *= cm_per_unit
        pts_cm = backproject_depth_map(depth_cm, cam, cam_from_world_cm, pixel_stride)
        all_pts_cm.append(pts_cm)
        frames_used += 1
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()
    print(f"  Back-projected {frames_used}/{len(images)} sampled frames", flush=True)

    if not all_pts_cm:
        return dict(error="no frames produced depth")
    pts_cm = np.concatenate(all_pts_cm, axis=0)
    n_raw = len(pts_cm)
    pts_cm = pts_cm[np.isfinite(pts_cm).all(axis=1)]
    if len(pts_cm) < n_raw:
        print(f"  Dropped {n_raw - len(pts_cm)} non-finite points (NaN/Inf from the depth "
              f"model on degenerate pixels — real, occasional occurrence on this footage)",
              flush=True)
    print(f"  Dense depth point cloud: {len(pts_cm)} points "
          f"(vs SfM's own {rec.num_points3D()} sparse points)", flush=True)
    if len(pts_cm) == 0:
        return dict(error="all backprojected points were non-finite")

    up = up_from_cameras(rec)
    R, z_floor = gravity_rotation_from_prior(up, pts_cm)
    if z_floor is None:
        return dict(error="could not localize the floor density peak")

    cam_centers_cm = np.array([im.projection_center() for im in rec.images.values()]) * cm_per_unit
    metrics, geometry = compute_room_dims(pts_cm, R, z_floor, cam_centers_cm)

    bound = ">= " if metrics["height_is_lower_bound"] else ""
    print(f"  Monocular-depth height : {bound}{metrics['height_cm']:.1f} cm"
          f"{'  [LOW CONFIDENCE]' if not metrics['height_reliable'] else ''}", flush=True)
    print(f"  Monocular-depth length : {metrics['length_cm']:.1f} cm", flush=True)
    print(f"  Monocular-depth width  : {metrics['width_cm']:.1f} cm", flush=True)

    recovered_ceiling = metrics["height_reliable"] and not metrics["height_is_lower_bound"]
    sfm_had_no_ceiling = existing is None or existing.get("height_is_lower_bound", True)
    verdict = "YES — recovers a ceiling SfM could not" if (recovered_ceiling and sfm_had_no_ceiling) \
        else ("no improvement over SfM" if not recovered_ceiling else "SfM already had a reliable ceiling")
    print(f"  Verdict: {verdict}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    viz_path = out_dir / "depth_room_topdown.png"
    render_topdown(geometry, R, z_floor, cm_per_unit, cam_centers_cm, None, viz_path)

    result = dict(cm_per_unit=cm_per_unit, source="monocular_depth", depth_model=depth_model_name,
                  frames_used=frames_used, n_points=len(pts_cm),
                  sfm_height_cm=existing.get("height_cm") if existing else None,
                  sfm_height_is_lower_bound=existing.get("height_is_lower_bound") if existing else None,
                  sfm_height_reliable=existing.get("height_reliable") if existing else None,
                  verdict=verdict, **metrics)
    out_path = out_dir / "depth_room_dims.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  Saved {out_path}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="Check whether monocular depth recovers the ceiling")
    parser.add_argument("--sfm-dir", type=Path)
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--model", default=None,
                        help="Sparse model subfolder name (default: auto-pick most-registered)")
    parser.add_argument("--sfm-root", type=Path,
                        help="Batch mode: root containing <video>/sfm + <video>/frames subfolders")
    parser.add_argument("--depth-model", default="depthanything_v2_metric",
                        choices=list(DEPTH_MODELS))
    parser.add_argument("--every-n-frames", type=int, default=5,
                        help="Depth inference is heavier than detection — sample, don't run on every frame")
    parser.add_argument("--pixel-stride", type=int, default=4,
                        help="Back-project every Nth pixel (4 = 1/16 of pixels) — dense enough, still tractable")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true",
                        help="Batch mode: redo videos that already have output (default: skip/resume)")
    args = parser.parse_args()
    if not args.sfm_dir and not args.sfm_root:
        parser.error("pass --sfm-dir (single video) or --sfm-root (batch)")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_fn, predict_fn = DEPTH_MODELS[args.depth_model]
    print(f"Loading {args.depth_model}...")
    loaded = load_fn(device)

    if args.sfm_dir:
        jobs = [(args.sfm_dir, args.frames_dir, args.out)]
    else:
        jobs = sorted(
            (p.parent / "sfm", p.parent / "frames", args.out / p.parent.name)
            for p in args.sfm_root.glob("*/sfm") if p.is_dir()
        )
        print(f"Batch mode: {len(jobs)} videos found under {args.sfm_root}")

    summary = []
    for sfm_dir, frames_dir, out_dir in jobs:
        video_name = out_dir.name
        existing_path = out_dir / "depth_room_dims.json"
        if args.sfm_root and not args.force and existing_path.exists():
            result = json.loads(existing_path.read_text())
            result["video"] = video_name
            summary.append(result)
            bound = ">=" if result.get("height_is_lower_bound") else ""
            print(f"\n=== {video_name} === (skipped, already done: "
                  f"H={bound}{result.get('height_cm', 0):.1f}cm, verdict={result.get('verdict')})",
                  flush=True)
            continue

        print(f"\n=== {video_name} ===", flush=True)
        model_name = args.model or best_sparse_model(sfm_dir)
        if model_name is None:
            result = dict(error="no sparse model found")
        else:
            try:
                result = process_video(sfm_dir, frames_dir, model_name, loaded, predict_fn, device,
                                       args.depth_model, args.every_n_frames, args.pixel_stride, out_dir)
            except Exception as exc:
                # one video's crash (bad frame, degenerate geometry, etc.) shouldn't
                # take down a multi-hour batch over the rest of the dataset
                result = dict(error=f"{type(exc).__name__}: {exc}")
            if device == "cuda":
                import torch
                torch.cuda.empty_cache()
        result["video"] = video_name
        summary.append(result)
        if "error" in result:
            print(f"  FAILED: {result['error']}", flush=True)

    if args.sfm_root:
        summary_path = args.out / "summary.json"
        args.out.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\nSaved batch summary: {summary_path}")
        print("\n" + "=" * 88)
        print(f"{'video':<45} {'SfM H(cm)':>12} {'Depth H(cm)':>12}  verdict")
        for r in summary:
            if "error" in r:
                print(f"{r['video']:<45} FAILED: {r['error']}")
            else:
                sfm_b = ">=" if r.get("sfm_height_is_lower_bound") else "  "
                dep_b = ">=" if r.get("height_is_lower_bound") else "  "
                sfm_h = f"{sfm_b}{r.get('sfm_height_cm'):.0f}" if r.get("sfm_height_cm") is not None else "n/a"
                print(f"{r['video']:<45} {sfm_h:>12} {dep_b}{r['height_cm']:>9.0f}  {r['verdict']}")


if __name__ == "__main__":
    main()
