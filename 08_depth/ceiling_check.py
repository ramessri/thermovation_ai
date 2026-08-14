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

Usage:
  python 08_depth/ceiling_check.py \
      --sfm-dir output/sfm/IMG_3126 --frames-dir dataset/frames/IMG_3126_sfm \
      --model 1 --depth-model depthanything_v2_metric \
      --out 08_depth/results/IMG_3126
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


def backproject_depth_map(depth_cm: np.ndarray, cam: pycolmap.Camera,
                          cam_from_world: np.ndarray, stride: int = 4) -> np.ndarray:
    """Every `stride`-th pixel of a metric depth map -> world-frame 3D points
    (model's own world frame, in CM — caller already has cm_per_unit-scaled
    camera poses, see main()).

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


def main():
    parser = argparse.ArgumentParser(description="Check whether monocular depth recovers the ceiling")
    parser.add_argument("--sfm-dir", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--model", default="1")
    parser.add_argument("--depth-model", default="depthanything_v2_metric",
                        choices=list(DEPTH_MODELS))
    parser.add_argument("--every-n-frames", type=int, default=5,
                        help="Depth inference is heavier than detection — sample, don't run on every frame")
    parser.add_argument("--pixel-stride", type=int, default=4,
                        help="Back-project every Nth pixel (4 = 1/16 of pixels) — dense enough, still tractable")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scale_info = json.loads((args.sfm_dir / "scale.json").read_text())
    cm_per_unit = scale_info["cm_per_unit"]
    rec = pycolmap.Reconstruction(str(args.sfm_dir / "sparse" / args.model))
    print(f"Model: {rec.num_reg_images()} images | scale {cm_per_unit:.3f} cm/unit")

    existing_dims_path = args.sfm_dir / "room_dims.json"
    existing = json.loads(existing_dims_path.read_text()) if existing_dims_path.exists() else None
    if existing:
        bound = ">=" if existing.get("height_is_lower_bound") else ""
        print(f"Existing SfM-only height: {bound}{existing.get('height_cm')} cm "
              f"(reliable={existing.get('height_reliable')})")

    load_fn, predict_fn = DEPTH_MODELS[args.depth_model]
    print(f"Loading {args.depth_model}...")
    loaded = load_fn(device)

    images = sorted(rec.images.values(), key=lambda im: im.name)[::args.every_n_frames]
    all_pts_cm = []
    frames_used = 0
    for img in images:
        bgr = cv2.imread(str(args.frames_dir / img.name))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cam = rec.cameras[img.camera_id]
        focal_px = float(cam.params[0])
        depth_cm = predict_fn(rgb, loaded, device, focal_px=focal_px)
        cam_from_world = np.asarray(img.cam_from_world().matrix())
        # camera center is in model units (arbitrary scale); depth is in cm.
        # Bridge via cm_per_unit: work entirely in cm by scaling the
        # translation component of the pose to cm before back-projecting.
        cam_from_world_cm = cam_from_world.copy()
        cam_from_world_cm[:, 3] *= cm_per_unit
        pts_cm = backproject_depth_map(depth_cm, cam, cam_from_world_cm, args.pixel_stride)
        all_pts_cm.append(pts_cm)
        frames_used += 1
    print(f"Back-projected {frames_used}/{len(images)} sampled frames")

    if not all_pts_cm:
        print("FAILED: no frames produced depth")
        return
    pts_cm = np.concatenate(all_pts_cm, axis=0)
    print(f"Dense depth point cloud: {len(pts_cm)} points "
          f"(vs SfM's own {rec.num_points3D()} sparse points)")

    up = up_from_cameras(rec)
    R, z_floor = gravity_rotation_from_prior(up, pts_cm)
    if z_floor is None:
        print("FAILED: could not localize the floor density peak in the depth point cloud")
        return

    cam_centers_cm = np.array([im.projection_center() for im in rec.images.values()]) * cm_per_unit
    metrics, geometry = compute_room_dims(pts_cm, R, z_floor, cam_centers_cm)

    bound = ">= " if metrics["height_is_lower_bound"] else ""
    print(f"\nMonocular-depth height : {bound}{metrics['height_cm']:.1f} cm"
          f"{'  [LOW CONFIDENCE]' if not metrics['height_reliable'] else ''}")
    print(f"Monocular-depth length : {metrics['length_cm']:.1f} cm")
    print(f"Monocular-depth width  : {metrics['width_cm']:.1f} cm")

    recovered_ceiling = metrics["height_reliable"] and not metrics["height_is_lower_bound"]
    sfm_had_no_ceiling = existing is None or existing.get("height_is_lower_bound", True)
    verdict = "YES — recovers a ceiling SfM could not" if (recovered_ceiling and sfm_had_no_ceiling) \
        else ("no improvement over SfM" if not recovered_ceiling else "SfM already had a reliable ceiling")
    print(f"\nVerdict: {verdict}")

    args.out.mkdir(parents=True, exist_ok=True)
    viz_path = args.out / "depth_room_topdown.png"
    render_topdown(geometry, R, z_floor, cm_per_unit, cam_centers_cm, None, viz_path)

    out = dict(cm_per_unit=cm_per_unit, source="monocular_depth", depth_model=args.depth_model,
              frames_used=frames_used, n_points=len(pts_cm),
              sfm_height_cm=existing.get("height_cm") if existing else None,
              sfm_height_is_lower_bound=existing.get("height_is_lower_bound") if existing else None,
              sfm_height_reliable=existing.get("height_reliable") if existing else None,
              verdict=verdict, **metrics)
    out_path = args.out / "depth_room_dims.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path}")
    print(f"Saved {viz_path}")


if __name__ == "__main__":
    main()
