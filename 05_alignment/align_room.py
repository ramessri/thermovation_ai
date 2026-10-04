"""
align_room.py — build a physically-anchored room coordinate frame for
top-down floor-plan inspection: Z = true vertical (gravity, from camera
poses, refined on floor inliers — same method as 05_geometry/room_dims.py),
origin = the floor point directly below the wall marker, Y = horizontal
direction from the marker into the room (toward the camera cluster), X
completes a right-handed frame (along the wall).

This differs from 05_alignment/align_world.py, which maps the marker
PLANE to world XY (Z = wall normal) — correct for placement work relative
to the wall, but a plain top-down (looking down Z) of that frame is a
face-on view of the wall, not a floor plan. Here Z is vertical, so a
top-down projection of this frame's output IS the bird's-eye room layout,
with the marker's wall as a fixed reference edge at the origin.

Requires marker_triangulation scale (scale.json must carry the marker's
own 9 triangulated 3D points) — not available for depth_ratio_fallback
scale (marker too small/distant to triangulate directly).

Usage:
  python 05_alignment/align_room.py output/pipeline/<video>/sfm
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "05_geometry"))
from room_dims import load_filtered_points, gravity_rotation, largest_cluster_mask

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align_world import pick_best_segment


def main():
    parser = argparse.ArgumentParser(
        description="Marker-anchored, gravity-up room frame + top-down floor plan")
    parser.add_argument("sfm_dir", type=Path,
                        help="SfM output dir containing sparse/<model> and scale.json")
    args = parser.parse_args()

    scale_info = json.loads((args.sfm_dir / "scale.json").read_text())
    segment = pick_best_segment(scale_info)
    if segment is None:
        print(f"FAILED: no marker segments in scale.json (method={scale_info.get('method')}) "
              "— this frame needs the marker's own triangulated points "
              "(marker_triangulation scale only, not depth_ratio_fallback).")
        return

    cm_per_unit = scale_info["cm_per_unit"]
    model_name = Path(scale_info["model_dir"]).name
    model_dir = args.sfm_dir / "sparse" / model_name
    rec = pycolmap.Reconstruction(str(model_dir))
    print(f"Model: {rec.num_reg_images()} images, {rec.num_points3D()} points, "
          f"scale {cm_per_unit:.4f} cm/unit")

    pts_cm = load_filtered_points(rec) * cm_per_unit
    R0, z_floor = gravity_rotation(rec, pts_cm)
    if z_floor is None:
        print("FAILED: could not localize the floor density peak")
        return
    print(f"Gravity-up rotation solved, floor at z={z_floor:.1f}cm (grav-aligned frame)")

    marker_pts_raw = np.array(list(segment["marker_points_model"].values()), dtype=np.float64)
    marker_center_1 = cm_per_unit * (R0 @ marker_pts_raw.mean(axis=0))

    cams_raw = np.array([im.projection_center() for im in rec.images.values()])
    cams_1 = cm_per_unit * (cams_raw @ R0.T)
    into_room = cams_1[:, :2].mean(axis=0) - marker_center_1[:2]
    into_room /= np.linalg.norm(into_room)
    world_x = np.array([into_room[1], -into_room[0]])   # right-handed with +Z up
    Raz = np.stack([world_x, into_room])                 # new_xy = Raz @ old_xy

    R_az3 = np.eye(3)
    R_az3[:2, :2] = Raz
    R_final = R_az3 @ R0
    t_xy = -Raz @ marker_center_1[:2]
    t_final = np.array([t_xy[0], t_xy[1], -z_floor])

    sim3 = pycolmap.Sim3d(cm_per_unit, pycolmap.Rotation3d(R_final), t_final)
    rec.transform(sim3)

    out_dir = args.sfm_dir / "aligned_room" / "sparse"
    out_dir.mkdir(parents=True, exist_ok=True)
    rec.write(str(out_dir))
    print(f"Saved room-frame aligned model to {out_dir}")
    print("Frame: origin = floor below marker, +X along wall, +Y into room, +Z up")

    # --- top-down floor plan: rec is now in the room frame, cm units ---
    pts = load_filtered_points(rec)
    xy_all = pts[:, :2]
    z_all = pts[:, 2]
    lo = np.percentile(xy_all, 1.0, axis=0)
    hi = np.percentile(xy_all, 99.0, axis=0)
    trimmed = np.all((xy_all >= lo) & (xy_all <= hi), axis=1)
    xy_trimmed = xy_all[trimmed].astype(np.float32)
    z_trimmed = z_all[trimmed]
    cluster_mask = largest_cluster_mask(xy_trimmed)
    xy_t = xy_trimmed[cluster_mask]
    z_t = z_trimmed[cluster_mask]
    (rcx, rcy), (rw, rh), angle = cv2.minAreaRect(xy_t.reshape(-1, 1, 2))
    length, width = max(rw, rh), min(rw, rh)
    height = float(np.percentile(z_t, 99) - 0.0)

    CANVAS, MARGIN = 900, 60
    span = max(hi[0] - lo[0], hi[1] - lo[1])
    s = (CANVAS - 2 * MARGIN) / span
    canvas = np.full((CANVAS, CANVAS, 3), 255, dtype=np.uint8)

    def to_px(p):
        return (int((p[0] - lo[0]) * s) + MARGIN,
                CANVAS - (int((p[1] - lo[1]) * s) + MARGIN))

    zn = np.clip(z_t / max(height, 1e-6), 0, 1)
    for p, t in zip(xy_t, zn):
        color = (int(200 * (1 - t) + 30), 60, int(200 * t + 30))   # blue=floor red=ceiling
        cv2.circle(canvas, to_px(p), 1, color, -1)
    box = cv2.boxPoints(((rcx, rcy), (rw, rh), angle))
    for i in range(4):
        cv2.line(canvas, to_px(box[i]), to_px(box[(i + 1) % 4]), (0, 160, 0), 2)
    centers = np.array([im.projection_center() for im in rec.images.values()])
    for c in centers:
        cv2.circle(canvas, to_px(c[:2]), 2, (0, 200, 255), -1)
    cv2.drawMarker(canvas, to_px((0, 0)), (255, 0, 255), cv2.MARKER_STAR, 22, 2)
    cv2.putText(canvas, f"L={length:.0f}cm  W={width:.0f}cm  (marker=magenta star, origin)",
                (MARGIN, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    viz_path = args.sfm_dir / "aligned_room" / "topdown.png"
    cv2.imwrite(str(viz_path), canvas)
    print(f"Saved {viz_path}")


if __name__ == "__main__":
    main()
