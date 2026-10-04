"""
gaussians_to_room_dims.py — room dimensions from a trained 3D Gaussian
Splatting model, using the exact same height/footprint algorithm as the
SfM arm (room_dims.py's compute_room_dims), so the two methods are judged
on identical math.

Filters the trained Gaussians by opacity (only confident, non-floater
splats) and physical size (reject huge low-density blobs — sky/background
artifacts, not real room surfaces) before treating their centers as a dense
point cloud, then reuses the ORIGINAL SfM camera poses (same scene, gsplat
was seeded from the same COLMAP model — see prepare_gsplat_dataset.py —
and should be trained with --pose_opt off and Parser normalize=False so
poses/scale stay identical). This isolates "does denser Gaussian geometry
recover more of the room" as the variable being measured, rather than
mixing in pose-optimization or rescaling effects too.

Usage:
  python 3d_recons_alternatives/gaussians_to_room_dims.py \
      --ply results/IMG_3126/ply/point_cloud_29999.ply \
      --sfm-dir output/sfm/IMG_3126 --model 1 \
      --out 3d_recons_alternatives/results/IMG_3126
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pycolmap
from plyfile import PlyData

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "05_geometry"))
from room_dims import gravity_rotation, compute_room_dims, render_topdown


def load_gaussian_centers(ply_path: Path, min_opacity: float, max_scale_cm: float,
                          cm_per_unit: float) -> tuple[np.ndarray, int]:
    """Gaussian means (model units) after filtering by opacity and physical
    size. PLY stores opacity as a logit and scale as log-scale (standard
    3DGS convention) — both need their inverse transform before filtering
    in real units. Returns (filtered_xyz, total_count_before_filtering)."""
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])],
                   axis=1).astype(np.float64)
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"])))
    scale = np.exp(np.stack([np.asarray(v["scale_0"]), np.asarray(v["scale_1"]),
                             np.asarray(v["scale_2"])], axis=1)).max(axis=1)
    scale_cm = scale * cm_per_unit

    keep = (opacity >= min_opacity) & (scale_cm <= max_scale_cm)
    print(f"Gaussians: {len(xyz)} total -> {int(keep.sum())} after "
          f"opacity>={min_opacity} and scale<={max_scale_cm}cm filter")
    return xyz[keep], len(xyz)


def main():
    parser = argparse.ArgumentParser(description="Room dimensions from a trained 3DGS model")
    parser.add_argument("--ply", type=Path, required=True, help="gsplat --save_ply output")
    parser.add_argument("--sfm-dir", type=Path, required=True,
                        help="The original SfM dir this 3DGS run was seeded from "
                             "(for camera poses + scale.json) — see prepare_gsplat_dataset.py")
    parser.add_argument("--model", default="1", help="Sparse model index used to seed gsplat")
    parser.add_argument("--min-opacity", type=float, default=0.5)
    parser.add_argument("--max-scale-cm", type=float, default=50.0,
                        help="Reject Gaussians physically larger than this (floaters/background blobs)")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output dir for gaussian_room_dims.json + gaussian_room_topdown.png")
    args = parser.parse_args()

    scale_info = json.loads((args.sfm_dir / "scale.json").read_text())
    cm_per_unit = scale_info["cm_per_unit"]
    rec = pycolmap.Reconstruction(str(args.sfm_dir / "sparse" / args.model))
    print(f"Reusing camera poses + scale from {args.sfm_dir} (model {args.model})")
    print(f"Scale: {cm_per_unit:.4f} cm/unit")

    xyz_units, n_total = load_gaussian_centers(args.ply, args.min_opacity,
                                               args.max_scale_cm, cm_per_unit)
    if len(xyz_units) < 100:
        print(f"FAILED: only {len(xyz_units)} Gaussians survived filtering — "
              "too sparse to compute room dimensions")
        return
    pts_cm = xyz_units * cm_per_unit

    R, z_floor = gravity_rotation(rec, pts_cm)
    if z_floor is None:
        print("FAILED: could not localize the floor density peak")
        return

    cam_centers_cm = np.array([img.projection_center() for img in rec.images.values()]) * cm_per_unit
    metrics, geometry = compute_room_dims(pts_cm, R, z_floor, cam_centers_cm)

    bound = ">= " if metrics["height_is_lower_bound"] else ""
    print(f"\nRoom height : {bound}{metrics['height_cm']:.1f} cm"
          f"{'  [LOW CONFIDENCE]' if not metrics['height_reliable'] else ''}")
    print(f"Room length : {metrics['length_cm']:.1f} cm"
          f"{'  [LOW CONFIDENCE]' if not metrics['footprint_reliable'] else ''}")
    print(f"Room width  : {metrics['width_cm']:.1f} cm")
    print(f"Points used (post-filter, post-gravity-fit): {metrics['points_used']}")

    args.out.mkdir(parents=True, exist_ok=True)
    viz_path = args.out / "gaussian_room_topdown.png"
    render_topdown(geometry, R, z_floor, cm_per_unit, cam_centers_cm,
                   scale_info.get("segments"), viz_path)

    out = dict(cm_per_unit=cm_per_unit, source="3dgs", ply=str(args.ply),
              n_gaussians_total=n_total, n_gaussians_kept=len(xyz_units),
              min_opacity=args.min_opacity, max_scale_cm=args.max_scale_cm,
              **metrics)
    out_path = args.out / "gaussian_room_dims.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path}")
    print(f"Saved {viz_path}")


if __name__ == "__main__":
    main()
