"""
room_dims.py — extract metric room dimensions (L x W x H) from a scaled
COLMAP sparse model.

Method:
  1. Filter the sparse cloud (track length, reprojection error, percentile trim).
  2. Estimate the up/gravity direction from camera poses (landscape phone
     video: the camera y-axis points roughly down), then refine it with a
     least-squares fit to the detected floor inliers.
  3. Floor and ceiling = strongest horizontal point-density peaks near the
     bottom / top of the height histogram → room height.
  4. Footprint = minimum-area rotated rectangle over the LARGEST CONNECTED
     CLUSTER of the top-down projection (rasterized + cv2.connectedComponents)
     — a plain percentile trim still lets a handful of disconnected outlier
     points (a separate reconstruction fragment, a stray far point) blow the
     box out to an implausible size, since minAreaRect fits whatever survives
     to the extremes.

All three outputs (height, footprint) carry an explicit *_reliable flag based
on minimum point-support thresholds — sparse/weak evidence is reported as
such instead of as a confident number.

Outputs <sfm_dir>/room_dims.json and a top-down floor-plan visualization.

Usage:
  python 05_geometry/room_dims.py output/sfm/<video_(sfm)>   # expects sparse/1 + scale.json inside
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap


def load_filtered_points(rec: pycolmap.Reconstruction,
                         min_track: int = 3, max_err_px: float = 1.5) -> np.ndarray:
    pts = [p.xyz for p in rec.points3D.values()
           if p.track.length() >= min_track and p.error < max_err_px]
    return np.asarray(pts, dtype=np.float64)


def up_from_cameras(rec: pycolmap.Reconstruction) -> np.ndarray:
    """Average camera y-axis (image 'down') in world coords → gravity prior."""
    downs = []
    for img in rec.images.values():
        R = np.asarray(img.cam_from_world().matrix())[:, :3]   # world->cam
        downs.append(R[1])           # cam y-axis expressed in world = row 1
    down = np.mean(downs, axis=0)
    up = -down / np.linalg.norm(down)
    return up


def rotation_to_z(up: np.ndarray) -> np.ndarray:
    """Rotation matrix taking `up` to +Z."""
    z = up / np.linalg.norm(up)
    x = np.cross([0.0, 1.0, 0.0], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([1.0, 0.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def largest_cluster_mask(xy: np.ndarray, cell_cm: float = 15.0,
                         dilate: int = 1) -> np.ndarray:
    """
    Boolean mask selecting the largest spatially-connected cluster of a
    top-down point set.

    Rasterizes points onto a `cell_cm`-sized occupancy grid and keeps only the
    points falling in the largest connected component (cv2.connectedComponents
    on the binary grid). This rejects disconnected outlier clusters / stray
    reconstruction fragments that would otherwise blow up a minAreaRect fit
    to an implausible size, without adding a new dependency (pure OpenCV).
    """
    lo = xy.min(axis=0)
    idx = np.floor((xy - lo) / cell_cm).astype(np.int32)
    w, h = int(idx[:, 0].max()) + 1, int(idx[:, 1].max()) + 1
    grid = np.zeros((h, w), dtype=np.uint8)
    grid[idx[:, 1], idx[:, 0]] = 255
    if dilate:
        grid = cv2.dilate(grid, np.ones((3, 3), np.uint8), iterations=dilate)
    n_labels, labels = cv2.connectedComponents(grid, connectivity=8)
    if n_labels <= 1:
        return np.ones(len(xy), dtype=bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0                                     # background
    best_label = int(np.argmax(counts))
    return labels[idx[:, 1], idx[:, 0]] == best_label


def density_peak(vals: np.ndarray, lo_pct: float, hi_pct: float,
                 bin_cm: float = 4.0) -> float | None:
    """Strongest histogram bin of `vals` restricted to a percentile band."""
    lo, hi = np.percentile(vals, [lo_pct, hi_pct])
    if hi - lo < bin_cm:
        return None
    bins = np.arange(lo, hi + bin_cm, bin_cm)
    counts, edges = np.histogram(vals[(vals >= lo) & (vals <= hi)], bins=bins)
    if counts.max() == 0:
        return None
    i = int(np.argmax(counts))
    return float((edges[i] + edges[i + 1]) / 2)


def gravity_rotation_from_prior(up_prior: np.ndarray, pts_cm: np.ndarray) -> tuple[np.ndarray, float | None]:
    """
    Rotation taking world -> gravity-aligned (+Z up), refined on floor inliers.

    Takes the up-vector prior directly rather than a pycolmap.Reconstruction,
    so it's reusable for any metric point cloud + prior — not just SfM's own
    (see gravity_rotation below for the SfM entry point, and
    3d_recons_alternatives/gaussians_to_room_dims.py for a 3DGS one).

    `pts_cm` are the filtered points already scaled to centimeters. Returns
    (R, z_floor) where z_floor is the floor height in the resulting frame
    (pts_cm @ R.T), or None if the floor density peak couldn't be localized.
    """
    up = up_prior
    for _ in range(2):
        R = rotation_to_z(up)
        z = (pts_cm @ R.T)[:, 2]
        z_floor = density_peak(z, 0.5, 35.0)
        if z_floor is None:
            break
        floor_pts = pts_cm[np.abs(z - z_floor) < 6.0]
        if len(floor_pts) < 50:
            break
        # least-squares plane normal of floor inliers
        c = floor_pts.mean(axis=0)
        _, _, vt = np.linalg.svd(floor_pts - c, full_matrices=False)
        n = vt[-1]
        up = n if n @ up > 0 else -n
    R = rotation_to_z(up)
    z_floor = density_peak((pts_cm @ R.T)[:, 2], 0.5, 35.0)
    return R, z_floor


def gravity_rotation(rec: pycolmap.Reconstruction, pts_cm: np.ndarray) -> tuple[np.ndarray, float | None]:
    """SfM entry point: gravity prior from camera poses, refined on floor inliers."""
    return gravity_rotation_from_prior(up_from_cameras(rec), pts_cm)


def compute_room_dims(pts_cm: np.ndarray, R: np.ndarray, z_floor: float,
                      cam_centers_cm: np.ndarray) -> tuple[dict, dict]:
    """
    Height (gap-based ceiling detection) + footprint (largest-connected-
    cluster min-area rect) from a gravity-aligned, scaled point cloud.

    Shared by every reconstruction-method arm (SfM here, 3D Gaussian
    Splatting in 3d_recons_alternatives/) so they're judged by identical
    math — a difference in the resulting numbers reflects the underlying
    geometry, not a divergence between two hand-rolled implementations.

    Returns (metrics, geometry): metrics is the JSON-safe dict shape
    room_dims.json has always had (minus cm_per_unit, which the caller
    already has and adds itself); geometry holds the numpy arrays a
    top-down render needs (see render_topdown).
    """
    ptsR = pts_cm @ R.T
    z = ptsR[:, 2]
    n_floor = int(np.sum(np.abs(z - z_floor) < 5))

    z_ceil, n_ceil, height, height_is_lower_bound = None, 0, None, False
    MIN_ABOVE, BIN, MIN_PTS, MIN_EXTENT = 90.0, 5.0, 15, 80.0
    GAP_CM, NOISE_FLOOR_PTS = 20.0, 2       # gap must be this tall & this empty
    h_rel = z - z_floor
    sel = h_rel > MIN_ABOVE
    if sel.sum() >= MIN_PTS:
        hh, xys = h_rel[sel], ptsR[sel][:, :2]
        edges = np.arange(MIN_ABOVE, hh.max() + BIN, BIN)
        counts = np.array([int(((hh >= edges[i]) & (hh < edges[i + 1])).sum())
                           for i in range(len(edges) - 1)])
        gap_bins = max(int(round(GAP_CM / BIN)), 1)

        # find the top of the "main mass": highest bin that is NOT preceded
        # by a full gap_bins-wide run of near-empty bins immediately below it
        mass_top = 0
        for i in range(len(counts)):
            if counts[i] > NOISE_FLOOR_PTS:
                mass_top = i
        # now scan bins strictly above mass_top for a candidate separated by
        # a genuine gap
        for i in range(mass_top + 1, len(counts)):
            if counts[i] < MIN_PTS:
                continue
            gap_start = i - gap_bins
            if gap_start <= mass_top:
                continue                                    # not enough room for a real gap
            if np.any(counts[gap_start:i] > NOISE_FLOOR_PTS):
                continue                                    # gap not actually empty
            m = (hh >= edges[i]) & (hh < edges[i + 1])
            ext = (np.percentile(xys[m], 97.5, axis=0)
                   - np.percentile(xys[m], 2.5, axis=0))
            if max(ext) >= MIN_EXTENT:
                z_ceil = float(z_floor + (edges[i] + edges[i + 1]) / 2)
                n_ceil = int(m.sum())
                break
    if z_ceil is not None:
        height = z_ceil - z_floor
    else:
        height = float(np.percentile(z, 99.5) - z_floor)
        height_is_lower_bound = True

    cam_z = (cam_centers_cm @ R.T)[:, 2]
    cam_height = float(np.median(cam_z) - z_floor)

    # footprint: percentile-trimmed top-down projection, largest connected
    # cluster only, then min-area rectangle
    MIN_FLOOR_PTS, MIN_CEIL_PTS, MIN_FOOTPRINT_PTS = 150, 80, 300
    xy_all = ptsR[:, :2]
    lo = np.percentile(xy_all, 1.0, axis=0)
    hi = np.percentile(xy_all, 99.0, axis=0)
    trimmed = np.all((xy_all >= lo) & (xy_all <= hi), axis=1)
    xy_trimmed = xy_all[trimmed].astype(np.float32)
    z_trimmed = z[trimmed]
    cluster_mask = largest_cluster_mask(xy_trimmed)
    xy_t = xy_trimmed[cluster_mask]
    z_t = z_trimmed[cluster_mask]
    rect = cv2.minAreaRect(xy_t.reshape(-1, 1, 2))
    (rcx, rcy), (rw, rh), angle = rect
    length, width = max(rw, rh), min(rw, rh)

    footprint_reliable = len(xy_t) >= MIN_FOOTPRINT_PTS
    height_reliable = (not height_is_lower_bound
                       and n_floor >= MIN_FLOOR_PTS and n_ceil >= MIN_CEIL_PTS)

    metrics = dict(
        height_cm=round(height, 1),
        height_is_lower_bound=height_is_lower_bound,
        height_reliable=height_reliable,
        length_cm=round(float(length), 1),
        width_cm=round(float(width), 1),
        footprint_reliable=footprint_reliable,
        floor_z_cm=round(z_floor, 1),
        ceiling_z_cm=None if z_ceil is None else round(z_ceil, 1),
        floor_peak_points=n_floor, ceiling_peak_points=n_ceil,
        camera_height_cm=round(cam_height, 1),
        points_used=int(len(xy_t)),
        points_trimmed=int(trimmed.sum()),
    )
    geometry = dict(lo=lo, hi=hi, xy_t=xy_t, z_t=z_t, rect=rect, height=height)
    return metrics, geometry


def render_topdown(geometry: dict, R: np.ndarray, z_floor: float, cm_per_unit: float,
                   cam_centers_cm: np.ndarray, marker_segments: list[dict] | None,
                   out_path: Path) -> None:
    """Top-down floor-plan visualization, shared across reconstruction methods."""
    lo, hi, xy_t, z_t = geometry["lo"], geometry["hi"], geometry["xy_t"], geometry["z_t"]
    (rcx, rcy), (rw, rh), angle = geometry["rect"]
    height = geometry["height"]

    CANVAS, MARGIN = 900, 60
    span = max(hi[0] - lo[0], hi[1] - lo[1])
    s = (CANVAS - 2 * MARGIN) / span
    canvas = np.full((CANVAS, CANVAS, 3), 255, dtype=np.uint8)

    def to_px(p):
        return (int((p[0] - lo[0]) * s) + MARGIN,
                CANVAS - (int((p[1] - lo[1]) * s) + MARGIN))

    zn = np.clip((z_t - z_floor) / max(height or 1.0, 1e-6), 0, 1)
    for p, t in zip(xy_t, zn):
        color = (int(200 * (1 - t) + 30), 60, int(200 * t + 30))   # blue=floor red=ceiling
        cv2.circle(canvas, to_px(p), 1, color, -1)
    box = cv2.boxPoints(((rcx, rcy), (rw, rh), angle))
    for i in range(4):
        cv2.line(canvas, to_px(box[i]), to_px(box[(i + 1) % 4]), (0, 160, 0), 2)
    centers = cam_centers_cm @ R.T
    for c in centers:
        cv2.circle(canvas, to_px(c[:2]), 2, (0, 200, 255), -1)
    for seg in (marker_segments or []):
        mp = np.array(list(seg["marker_points_model"].values()), dtype=np.float64)
        mc = (mp.mean(axis=0) * cm_per_unit) @ R.T
        cv2.drawMarker(canvas, to_px(mc[:2]), (255, 0, 255), cv2.MARKER_STAR, 18, 2)
    cv2.putText(canvas, f"L={rw if rw>rh else rh:.0f}cm  W={rw if rw<rh else rh:.0f}cm  H={height:.0f}cm",
                (MARGIN, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite(str(out_path), canvas)


def main():
    parser = argparse.ArgumentParser(description="Room dimensions from scaled SfM model")
    parser.add_argument("sfm_dir", type=Path,
                        help="SfM output dir containing sparse/<i> and scale.json")
    parser.add_argument("--model", default="1", help="Sparse model index (default 1)")
    parser.add_argument("--scale-json", type=Path, default=None,
                        help="Override which scale file to use (default: <sfm_dir>/scale.json, "
                             "the marker/Zhang-derived one). Point this at e.g. "
                             "<sfm_dir>/scale_monocular_depth_<model>.json (see "
                             "08_depth/depth_scale.py) to compute room dimensions from the SAME "
                             "SfM point cloud but a depth-model-derived scale instead — isolates "
                             "whether the scale source or the geometry is doing the work. Output "
                             "then goes to room_dims_<scale-file-stem>.json instead of "
                             "room_dims.json, so it never overwrites the marker-based result.")
    args = parser.parse_args()

    scale_path = args.scale_json or (args.sfm_dir / "scale.json")
    scale_info = json.loads(scale_path.read_text())
    cm_per_unit = scale_info["cm_per_unit"]
    rec = pycolmap.Reconstruction(str(args.sfm_dir / "sparse" / args.model))
    print(f"Model: {rec.num_reg_images()} images, {rec.num_points3D()} points")
    print(f"Scale: {cm_per_unit:.4f} cm/unit (from {scale_path.name}, method={scale_info.get('method')})")

    pts = load_filtered_points(rec) * cm_per_unit
    print(f"Filtered points: {len(pts)}")

    # align gravity to +Z (prior from cameras, refined on floor inliers)
    R, z_floor = gravity_rotation(rec, pts)
    if z_floor is None:
        print("FAILED: could not localize the floor density peak")
        return

    cam_centers_cm = np.array([img.projection_center() for img in rec.images.values()]) * cm_per_unit
    metrics, geometry = compute_room_dims(pts, R, z_floor, cam_centers_cm)

    if metrics["height_is_lower_bound"]:
        print("WARNING: no gap-separated ceiling band found (topmost points "
              "are contiguous with the general equipment/pipe mass, not a "
              "distinct plane) — height reported as a LOWER BOUND. "
              "Re-film including floor-wall-ceiling junctions (see protocol).")

    bound = ">= " if metrics["height_is_lower_bound"] else ""
    flag_h = "" if metrics["height_reliable"] else "  [LOW CONFIDENCE]"
    flag_f = "" if metrics["footprint_reliable"] else "  [LOW CONFIDENCE]"
    _, _, angle = geometry["rect"]
    print(f"\nRoom height : {bound}{metrics['height_cm']:.1f} cm   "
          f"(floor peak {metrics['floor_peak_points']} pts, "
          f"ceiling peak {metrics['ceiling_peak_points']} pts){flag_h}")
    print(f"Room length : {metrics['length_cm']:.1f} cm{flag_f}")
    print(f"Room width  : {metrics['width_cm']:.1f} cm{flag_f}")
    print(f"Camera height above floor (median): {metrics['camera_height_cm']:.0f} cm")
    print(f"Footprint rectangle angle: {angle:.1f} deg")
    print(f"Footprint cluster size: {metrics['points_used']} / {metrics['points_trimmed']} "
          f"trimmed points (largest connected component)")
    if not metrics["footprint_reliable"]:
        print(f"WARNING: footprint support ({metrics['points_used']} pts) below the "
              f"300-point confidence threshold — L/W likely unreliable.")
    if not metrics["height_reliable"] and not metrics["height_is_lower_bound"]:
        print(f"WARNING: floor/ceiling support (floor={metrics['floor_peak_points']}, "
              f"ceiling={metrics['ceiling_peak_points']}) below confidence thresholds "
              "(150/80) — height likely unreliable despite a numeric ceiling match.")

    if args.scale_json:
        dims_name = f"room_dims_{scale_path.stem}.json"
        viz_name = f"room_topdown_{scale_path.stem}.png"
    else:
        dims_name, viz_name = "room_dims.json", "room_topdown.png"

    viz_path = args.sfm_dir / viz_name
    render_topdown(geometry, R, z_floor, cm_per_unit, cam_centers_cm,
                   scale_info.get("segments"), viz_path)

    out = dict(cm_per_unit=cm_per_unit, scale_source=scale_info.get("method"), **metrics)
    dims_path = args.sfm_dir / dims_name
    dims_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {dims_path}")
    print(f"Saved {viz_path}")


if __name__ == "__main__":
    main()
