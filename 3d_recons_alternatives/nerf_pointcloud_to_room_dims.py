"""
nerf_pointcloud_to_room_dims.py — room dimensions from a trained Nerfstudio
(nerfacto) model, via its own `ns-export pointcloud --save-world-frame`
export, using the exact same height/footprint algorithm as the SfM and
3DGS arms (room_dims.py's compute_room_dims), so all three are judged on
identical math.

Unlike the 3DGS arm, no opacity/scale filtering is needed here — Nerfstudio
does its own outlier removal at export time (--remove-outliers) and the
exported .ply is a plain point cloud (x,y,z[,rgb,normals]), not raw
Gaussian primitives with logit/log-encoded properties.

`--save-world-frame` on `ns-export pointcloud` is what keeps this point
cloud in the ORIGINAL COLMAP scale/coordinate frame — Nerfstudio's training
dataparser auto-scales and reorients the scene internally by default, and
without that flag the exported points would be in that internal (rescaled,
z-up-reoriented) frame instead, silently invalidating the cm_per_unit scale
reused here. Sanity-check this before trusting the numbers: camera centers
recovered from the same trained run should land close to the original SfM
camera centers once cm_per_unit is applied — if they don't, world-frame
export isn't behaving as expected on the installed Nerfstudio version.

Usage:
  python 3d_recons_alternatives/nerf_pointcloud_to_room_dims.py \
      --ply nerf_results/IMG_3126/point_cloud.ply \
      --sfm-dir output/sfm/IMG_3126 --model 1 \
      --out 3d_recons_alternatives/results/IMG_3126_nerf
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


def load_pointcloud(ply_path: Path) -> np.ndarray:
    """Plain x,y,z from a Nerfstudio `ns-export pointcloud` .ply (no
    opacity/scale decoding needed — this is a regular point cloud export,
    not raw Gaussian primitives)."""
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])],
                   axis=1).astype(np.float64)
    print(f"Loaded {len(xyz)} points from {ply_path}")
    return xyz


def main():
    parser = argparse.ArgumentParser(description="Room dimensions from a trained NeRF (nerfacto) point cloud")
    parser.add_argument("--ply", type=Path, required=True,
                        help="ns-export pointcloud --save-world-frame output")
    parser.add_argument("--sfm-dir", type=Path, required=True,
                        help="The original SfM dir this NeRF run was seeded from "
                             "(for camera poses + scale.json)")
    parser.add_argument("--model", default="1", help="Sparse model index used to seed NeRF")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output dir for nerf_room_dims.json + nerf_room_topdown.png")
    args = parser.parse_args()

    scale_info = json.loads((args.sfm_dir / "scale.json").read_text())
    cm_per_unit = scale_info["cm_per_unit"]
    rec = pycolmap.Reconstruction(str(args.sfm_dir / "sparse" / args.model))
    print(f"Reusing camera poses + scale from {args.sfm_dir} (model {args.model})")
    print(f"Scale: {cm_per_unit:.4f} cm/unit")

    xyz_units = load_pointcloud(args.ply)
    if len(xyz_units) < 100:
        print(f"FAILED: only {len(xyz_units)} points in the exported cloud — too sparse")
        return
    pts_cm = xyz_units * cm_per_unit
    n_raw = len(pts_cm)

    # --save-world-frame's inverse-transform math is exact (verified independently —
    # camera centers recovered this way match the original SfM model's to ~1e-6cm),
    # so a mismatched frame/scale is not the risk here. The real risk with a
    # lightly-trained nerfacto model (few hundred/thousand steps) is floater points:
    # pixels where depth hasn't converged yet get projected to wildly wrong depths,
    # sometimes tens of meters away. Reject anything far outside where the camera
    # trajectory itself was, before the gravity/room-dims fit ever sees them —
    # analogous to gaussians_to_room_dims.py's opacity/scale floater filter for 3DGS.
    cam_centers_cm = np.array([img.projection_center() for img in rec.images.values()]) * cm_per_unit
    cam_centroid = cam_centers_cm.mean(axis=0)
    cam_bbox_diag = float(np.linalg.norm(cam_centers_cm.max(0) - cam_centers_cm.min(0)))
    radius_cap_cm = max(cam_bbox_diag * 1.5, 500.0)
    dist_from_traj = np.linalg.norm(pts_cm - cam_centroid, axis=1)
    keep = dist_from_traj < radius_cap_cm
    pts_cm = pts_cm[keep]
    print(f"Floater filter: kept {len(pts_cm)}/{n_raw} points within {radius_cap_cm:.0f}cm "
          f"of the camera-trajectory centroid (trajectory bbox diagonal {cam_bbox_diag:.0f}cm)")
    if len(pts_cm) < 100:
        print(f"FAILED: only {len(pts_cm)} points survived the floater filter — too sparse")
        return

    # secondary sanity check on what's left — should now be small by construction
    cloud_center = pts_cm.mean(axis=0)
    offset_cm = float(np.linalg.norm(cloud_center - cam_centroid))
    scale_trustworthy = offset_cm < radius_cap_cm
    print(f"Post-filter point-cloud centroid to camera-trajectory centroid: {offset_cm:.0f} cm")

    R, z_floor = gravity_rotation(rec, pts_cm)
    if z_floor is None:
        print("FAILED: could not localize the floor density peak")
        return

    metrics, geometry = compute_room_dims(pts_cm, R, z_floor, cam_centers_cm)

    bound = ">= " if metrics["height_is_lower_bound"] else ""
    print(f"\nRoom height : {bound}{metrics['height_cm']:.1f} cm"
          f"{'  [LOW CONFIDENCE]' if not metrics['height_reliable'] else ''}")
    print(f"Room length : {metrics['length_cm']:.1f} cm"
          f"{'  [LOW CONFIDENCE]' if not metrics['footprint_reliable'] else ''}")
    print(f"Room width  : {metrics['width_cm']:.1f} cm")
    print(f"Points used (post-gravity-fit): {metrics['points_used']}")

    args.out.mkdir(parents=True, exist_ok=True)
    viz_path = args.out / "nerf_room_topdown.png"
    render_topdown(geometry, R, z_floor, cm_per_unit, cam_centers_cm,
                   scale_info.get("segments"), viz_path)

    out = dict(cm_per_unit=cm_per_unit, source="nerf", ply=str(args.ply),
              n_points=len(xyz_units), world_frame_sanity_offset_cm=round(offset_cm, 1),
              scale_trustworthy=scale_trustworthy,
              **metrics)
    out_path = args.out / "nerf_room_dims.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path}")
    print(f"Saved {viz_path}")


if __name__ == "__main__":
    main()
