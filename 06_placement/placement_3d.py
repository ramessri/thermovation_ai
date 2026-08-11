"""
placement_3d.py — 3D placement recommendation on the marker wall.

Uses everything the pipeline has already solved:
  - the triangulated marker (scale.json) defines the candidate WALL PLANE
    and the metric scale,
  - the sparse cloud provides obstacles near that wall,
  - the Rücklauf is located via the blue-marking convention (blue cap/ring
    = return flow) and lifted to 3D through its co-visible feature points,
  - the floor height (room_dims logic) constrains the mounting band.

Candidate unit rectangles are scored in real cm:
  score = w_dist * (3D distance center→Rücklauf)
        + w_clear * max(0, min_clearance - nearest_obstacle_distance)
        + w_height * |mount_height - preferred_height|

Outputs: placement_3d.json + overlay rendering on the best marker frame.

Usage:
  python 06_placement/placement_3d.py output/sfm/IMG_3126 dataset/frames/IMG_3126_sfm
  # optional manual Rücklauf override if the blue cue fails:
  #   --rucklauf-frame IMG_3126_0004.jpg --rucklauf-px 640 360
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
sys.path.insert(0, str(_ROOT / "04_scale"))
from room_dims import load_filtered_points, up_from_cameras, rotation_to_z, density_peak
from scale_sfm import detect_in_registered_frames

INVALID_P3D = 2**63 - 1


# ── Rücklauf localization (blue-marking cue) ─────────────────────────────────

BLUE_HSV_LOW = (95, 90, 60)
BLUE_HSV_HIGH = (130, 255, 255)


def blue_circle_candidates(bgr: np.ndarray) -> list[dict]:
    """Small blue circular caps/rings (German convention: blue = return flow).

    Large blue objects (tanks, vessels) are rejected via a size cap and an
    annulus check: a valve cap sits on metal/pipe, so its surroundings must
    NOT be blue.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_HSV_LOW, BLUE_HSV_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    img_area = bgr.shape[0] * bgr.shape[1]
    h, w = mask.shape
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 150 or area > 0.005 * img_area:      # caps are small
            continue
        per = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (per * per + 1e-9)
        if circularity < 0.55:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        # annulus check: ring at 1.4-2.0r around the cap must be mostly non-blue
        y0, y1 = max(int(cy - 2 * r), 0), min(int(cy + 2 * r), h)
        x0, x1 = max(int(cx - 2 * r), 0), min(int(cx + 2 * r), w)
        patch = mask[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        rr = np.hypot(xx - cx, yy - cy)
        annulus = (rr > 1.4 * r) & (rr < 2.0 * r)
        if annulus.sum() == 0 or patch[annulus].mean() > 0.3 * 255:
            continue                                    # embedded in a big blue object
        out.append(dict(px=float(cx), py=float(cy), r=float(r),
                        area=area, circularity=circularity,
                        strength=float(area * circularity)))
    return out


def depth_from_covisible(rec: pycolmap.Reconstruction, img, px: float, py: float,
                         radius_px: float = 60.0) -> np.ndarray | None:
    """3D position from feature points observed near a pixel in this image."""
    xyz = []
    for p2d in img.points2D:
        pid = p2d.point3D_id
        if pid == INVALID_P3D or pid not in rec.points3D:
            continue
        if np.hypot(p2d.xy[0] - px, p2d.xy[1] - py) < radius_px:
            xyz.append(rec.points3D[pid].xyz)
    if len(xyz) < 5:
        return None
    return np.median(np.asarray(xyz), axis=0)


def locate_rucklauf(rec: pycolmap.Reconstruction, frames_dir: Path,
                    override: tuple[str, float, float] | None,
                    cm_per_unit: float,
                    pipe_candidate: dict | None = None,
                    agree_threshold_cm: float = 30.0) -> tuple[np.ndarray, str | None, tuple | None, str]:
    """Best Rücklauf 3D estimate, combining the blue-cap cue with the
    pipe-color-pairing signal (07_pipes/pipe_paths.py) when available.

    Returns (X, frame_name_or_None, px_py_or_None, confidence), where
    confidence is 'manual' (explicit override), 'high' (both signals agree
    within agree_threshold_cm), 'medium' (only one signal available), or
    'low' (both available but disagree — blue-cap result is used, since a
    successful circular-cap detection is the stronger single cue, but the
    disagreement is reported rather than hidden).
    """
    if override:
        name, px, py = override
        img = next(im for im in rec.images.values() if im.name == name)
        X = depth_from_covisible(rec, img, px, py)
        if X is None:
            raise RuntimeError("No co-visible 3D points near the manual Rücklauf pixel")
        return X, name, (px, py), "manual"

    blue_result = None
    for img in sorted(rec.images.values(), key=lambda im: im.name):
        bgr = cv2.imread(str(frames_dir / img.name))
        if bgr is None:
            continue
        for c in blue_circle_candidates(bgr):
            X = depth_from_covisible(rec, img, c["px"], c["py"])
            if X is None:
                continue
            if blue_result is None or c["strength"] > blue_result[0]:
                blue_result = (c["strength"], X, img.name, (c["px"], c["py"]))

    pipe_X = None
    if pipe_candidate and pipe_candidate.get("rucklauf_xyz_model") is not None:
        pipe_X = np.array(pipe_candidate["rucklauf_xyz_model"])

    if blue_result is not None and pipe_X is not None:
        _, bx, bname, bpx = blue_result
        agree_cm = float(np.linalg.norm(bx - pipe_X)) * cm_per_unit
        if agree_cm < agree_threshold_cm:
            print(f"Rücklauf cue: {bname} at ({bpx[0]:.0f},{bpx[1]:.0f}) — "
                  f"blue-cap and pipe-pairing agree within {agree_cm:.0f}cm, high confidence")
            return bx, bname, bpx, "high"
        print(f"Rücklauf: blue-cap ({bname}) and pipe-pairing DISAGREE by "
              f"{agree_cm:.0f}cm — low confidence, using blue-cap")
        return bx, bname, bpx, "low"

    if blue_result is not None:
        _, bx, bname, bpx = blue_result
        print(f"Rücklauf cue: {bname} at ({bpx[0]:.0f},{bpx[1]:.0f}) "
              "[blue-cap only, medium confidence]")
        return bx, bname, bpx, "medium"

    if pipe_X is not None:
        print("Rücklauf: pipe-color-pairing only (no blue-cap cue found), medium confidence")
        return pipe_X, None, None, "medium"

    raise RuntimeError(
        "No blue Rücklauf cue and no pipe-pairing candidate found. "
        "Provide --rucklauf-frame/--rucklauf-px manually, or run 07_pipes/pipe_paths.py first."
    )


# ── wall frame from the triangulated marker ──────────────────────────────────

def _axes_from_normal(origin, n, up_world):
    """In-plane axes: u horizontal, v as vertical as possible; normal faces up-ish v."""
    u = np.cross(up_world, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    if v @ up_world < 0:
        v, u = -v, -u
    return origin, u, v, n


def marker_wall_frame(scale_info: dict, up_world: np.ndarray):
    """Wall plane (model units) from the triangulated marker (precise path)."""
    seg = max(scale_info["segments"], key=lambda s: s["n_views"])
    P9 = np.array(list(seg["marker_points_model"].values()), dtype=np.float64)
    origin = P9.mean(axis=0)
    _, _, vt = np.linalg.svd(P9 - origin)
    return _axes_from_normal(origin, vt[-1], up_world)


def wall_frame_ransac(pts_model, up_world, ref_model, z_floor_cm, cm_per_unit,
                      iters: int = 3000, thresh_cm: float = 6.0, near_cm: float = 250.0):
    """
    Find the dominant VERTICAL wall plane near a reference point (the Rücklauf).

    Works for videos without a triangulated marker and with the marker on
    multiple surfaces: instead of trusting the marker, we RANSAC a wall line in
    the gravity-aligned top-down projection, restricted to points near the
    Rücklauf and within a wall height band. The plane is vertical by
    construction (normal horizontal), so the tilt guard always passes.
    Returns (origin, u, v, n) in model frame, plus inlier count.
    """
    R = rotation_to_z(up_world)
    q = pts_model @ R.T                          # gravity-aligned, model units
    zc = q[:, 2] * cm_per_unit
    band = (zc > z_floor_cm + 15) & (zc < z_floor_cm + 230)   # wall band
    qb = q[band]
    if len(qb) < 50:
        return None
    ref_q = ref_model @ R.T
    d_top = np.hypot((qb[:, 0] - ref_q[0]) * cm_per_unit,
                     (qb[:, 1] - ref_q[1]) * cm_per_unit)
    pool = qb[d_top < near_cm]
    if len(pool) < 50:
        pool = qb
    xy = pool[:, :2]
    t = thresh_cm / cm_per_unit
    rng = np.random.default_rng(0)
    N = len(xy)
    best_inl, best = 0, None
    for _ in range(iters):
        i, j = rng.integers(0, N, 2)
        e = xy[j] - xy[i]
        L = np.linalg.norm(e)
        if L < 1e-6:
            continue
        e = e / L
        nrm = np.array([-e[1], e[0]])
        inl = np.abs((xy - xy[i]) @ nrm) < t
        c = int(inl.sum())
        if c > best_inl:
            best_inl, best = c, inl
    if best is None or best_inl < 30:
        return None
    ptsin = xy[best]
    c0 = ptsin.mean(0)
    _, _, vt = np.linalg.svd(ptsin - c0)
    e = vt[0]
    nrm = np.array([-e[1], e[0]])
    uq = np.array([e[0], e[1], 0.0])
    vq = np.array([0.0, 0.0, 1.0])
    nq = np.array([nrm[0], nrm[1], 0.0])
    oq = np.array([c0[0], c0[1], z_floor_cm / cm_per_unit])
    origin = R.T @ oq
    u = R.T @ uq
    v = R.T @ vq
    n = R.T @ nq
    return origin, u, v, n, best_inl


def wall_frame_from_sightings(rec, sightings, up_world):
    """
    Wall plane from SfM points inside the marker footprint, fitted robustly.

    Used when the marker itself couldn't be triangulated (small/distant marker,
    depth-ratio scale). The marker lies flat on the wall, so well-triangulated
    SfM points projecting inside its footprint define the wall plane directly.
    Returns (origin, u, v, n, marker_center) or None.
    """
    name_to_img = {im.name: im for im in rec.images.values()}
    xyz = []
    for s in sightings:
        img = name_to_img[s["name"]]
        mcx, mcy = s["pts"][4]
        radius = 0.6 * float(np.ptp(s["pts"], axis=0).max())
        for p2d in img.points2D:
            pid = p2d.point3D_id
            if pid == INVALID_P3D or pid not in rec.points3D:
                continue
            if np.hypot(p2d.xy[0] - mcx, p2d.xy[1] - mcy) < radius:
                xyz.append(rec.points3D[pid].xyz)
    if len(xyz) < 20:
        return None
    P = np.asarray(xyz)
    origin = P.mean(axis=0)
    # robust plane: fit, reject far points, refit
    for _ in range(3):
        _, _, vt = np.linalg.svd(P - origin)
        n = vt[-1]
        resid = np.abs((P - origin) @ n)
        keep = resid < max(2.5 * np.median(resid), 1e-6)
        if keep.sum() < 20:
            break
        P = P[keep]
        origin = P.mean(axis=0)
    origin, u, v, n = _axes_from_normal(origin, n, up_world)
    return origin, u, v, n, P.mean(axis=0)


def main():
    parser = argparse.ArgumentParser(description="3D placement recommendation on the marker wall")
    parser.add_argument("sfm_dir", type=Path)
    parser.add_argument("frames_dir", type=Path)
    parser.add_argument("--model", default="1")
    parser.add_argument("--unit-wh-cm", type=float, nargs=2, default=[60.0, 40.0],
                        help="Indoor unit width x height in cm (default 60 40)")
    parser.add_argument("--min-clearance-cm", type=float, default=10.0)
    parser.add_argument("--preferred-height-cm", type=float, default=120.0,
                        help="Preferred mounting height of unit center above floor")
    parser.add_argument("--min-bottom-cm", type=float, default=20.0,
                        help="Minimum height of the unit's bottom edge above the floor")
    parser.add_argument("--w-dist", type=float, default=1.0)
    parser.add_argument("--w-clear", type=float, default=3.0)
    parser.add_argument("--w-height", type=float, default=0.5)
    parser.add_argument("--rucklauf-frame", default=None)
    parser.add_argument("--rucklauf-px", type=float, nargs=2, default=None)
    args = parser.parse_args()

    scale_info = json.loads((args.sfm_dir / "scale.json").read_text())
    cm_per_unit = scale_info["cm_per_unit"]
    rec = pycolmap.Reconstruction(str(args.sfm_dir / "sparse" / args.model))
    print(f"Model: {rec.num_reg_images()} images | scale {cm_per_unit:.3f} cm/unit")

    up = up_from_cameras(rec)

    # floor height (same approach as room_dims)
    pts_cm = load_filtered_points(rec) * cm_per_unit
    R_up = rotation_to_z(up)
    z_all = (pts_cm @ R_up.T)[:, 2]
    z_floor = density_peak(z_all, 0.5, 35.0)
    print(f"Floor at z={z_floor:.1f} cm (gravity-aligned)")

    # Rücklauf 3D (located first — the fallback wall search keys off it)
    override = None
    if args.rucklauf_frame and args.rucklauf_px:
        override = (args.rucklauf_frame, args.rucklauf_px[0], args.rucklauf_px[1])
    pipe_lengths_path = args.sfm_dir / "pipe_lengths.json"
    pipe_candidate = None
    if pipe_lengths_path.exists():
        pipe_data = json.loads(pipe_lengths_path.read_text())
        pipe_candidate = pipe_data.get("rucklauf_pipe_pairing")
    ruck_X, ruck_frame, ruck_px, ruck_confidence = locate_rucklauf(
        rec, args.frames_dir, override, cm_per_unit, pipe_candidate)

    # wall plane: triangulated marker if available, else RANSAC the vertical wall
    # nearest the Rücklauf (robust when the marker sits on multiple surfaces)
    sightings = None
    if scale_info.get("segments"):
        origin, u_ax, v_ax, n_ax = marker_wall_frame(scale_info, up)
        wall_source = "triangulated marker"
    else:
        sightings = detect_in_registered_frames(rec, args.frames_dir)
        pts_model = load_filtered_points(rec)
        wf = wall_frame_ransac(pts_model, up, ruck_X, z_floor, cm_per_unit)
        if wf is None:
            print("FAILED: no dominant vertical wall found near the Rücklauf")
            return
        origin, u_ax, v_ax, n_ax, n_inl = wf
        wall_source = f"RANSAC vertical wall near Rücklauf ({n_inl} inliers)"
    wall_tilt = float(np.degrees(np.arcsin(abs(n_ax @ up))))
    print(f"Wall plane [{wall_source}]: origin={np.round(origin, 3)}, "
          f"tilt from vertical: {wall_tilt:.1f} deg")
    if wall_tilt > 25.0:
        print(f"FAILED: wall plane is {wall_tilt:.0f}° from vertical — unreliable; skipping.")
        return

    ruck_dist_wall = float((ruck_X - origin) @ n_ax) * cm_per_unit
    print(f"Rücklauf 3D (model units): {np.round(ruck_X, 3)}  "
          f"({abs(ruck_dist_wall):.0f} cm from wall plane)")

    # wall coordinate grid: project cloud points near the wall plane
    pts = np.asarray([p.xyz for p in rec.points3D.values()
                      if p.track.length() >= 3 and p.error < 1.5])
    d_plane = (pts - origin) @ n_ax * cm_per_unit          # signed cm from plane
    uv = np.stack([(pts - origin) @ u_ax, (pts - origin) @ v_ax], axis=1) * cm_per_unit

    # obstacles only on the camera side of the wall
    cams = np.array([im.projection_center() for im in rec.images.values()])
    front_sign = np.sign(float(np.median((cams - origin) @ n_ax)))
    d_front = d_plane * front_sign

    def density_filter(pts_uv: np.ndarray, radius: float = 6.0,
                       min_neighbors: int = 3) -> np.ndarray:
        """Keep points with enough neighbors — lone points are depth noise."""
        if len(pts_uv) < 2:
            return pts_uv
        keep = np.zeros(len(pts_uv), dtype=bool)
        for i in range(0, len(pts_uv), 512):
            chunk = pts_uv[i:i + 512]
            dist = np.linalg.norm(chunk[:, None, :] - pts_uv[None, :, :], axis=2)
            keep[i:i + 512] = (dist < radius).sum(axis=1) >= min_neighbors
        return pts_uv[keep]

    wall_band = np.abs(d_plane) < 8.0                       # wall surface points
    obst_band = (d_front >= 8.0) & (d_front < 40.0)         # stuff in front of wall
    uv_wall = uv[wall_band]
    uv_obst = density_filter(uv[obst_band])
    # Window/recess keep-out is only meaningful when the plane IS the actual wall
    # surface (triangulated marker). For a RANSAC plane sliced through the cloud,
    # points behind it are mostly deeper wall, not recesses — so skip it there.
    if scale_info.get("segments"):
        hole_band = (d_front <= -8.0) & (d_front > -80.0)   # recesses: windows/niches
        uv_hole = density_filter(uv[hole_band])
    else:
        uv_hole = np.empty((0, 2))
    print(f"Wall-surface points: {len(uv_wall)}   obstacle points: {len(uv_obst)}   "
          f"hole/recess points (window keep-out): {len(uv_hole)}")
    if len(uv_wall) < 100:
        print("WARNING: sparse wall coverage — placement area may be underestimated")

    # usable wall extent: the wall physically continues under obstacle-covered
    # areas, so take the union of ALL near-wall evidence (wall surface,
    # obstacles, recess rims), not just bare wall-surface points — otherwise
    # regions close to the Rücklauf are never searched.
    uv_evidence = np.concatenate([uv_wall, uv_obst] + ([uv_hole] if len(uv_hole) else []))
    lo = np.percentile(uv_evidence, 1, axis=0)
    hi = np.percentile(uv_evidence, 99, axis=0)

    # floor height in wall coords: v of a point at floor level
    origin_z = float(((origin * cm_per_unit) @ R_up.T)[2])
    v_floor = z_floor - origin_z                            # cm, along v (≈vertical)

    # Rücklauf in wall coordinates (for reporting/search-extent sanity)
    ruck_uv = np.array([float((ruck_X - origin) @ u_ax),
                        float((ruck_X - origin) @ v_ax)]) * cm_per_unit
    print(f"Wall extent u: [{lo[0]:.0f}, {hi[0]:.0f}] cm  "
          f"v: [{lo[1]:.0f}, {hi[1]:.0f}] cm  floor at v={v_floor:.0f} cm")
    print(f"Rücklauf in wall coords: u={ruck_uv[0]:.0f} cm, v={ruck_uv[1]:.0f} cm")

    # candidate grid
    W, H = args.unit_wh_cm
    step = 5.0
    cands = []
    n_cells = n_hole_rej = n_obst_rej = 0
    for cu in np.arange(lo[0] + W / 2, hi[0] - W / 2 + 1e-6, step):
        for cv_ in np.arange(max(lo[1] + H / 2, v_floor + args.min_bottom_cm + H / 2),
                             hi[1] - H / 2 + 1e-6, step):
            # NOTE: no wall-point support requirement — blank (featureless) wall
            # is exactly where placement is possible; the plane itself is known
            # precisely from the marker. Extent is bounded by [lo, hi] above.
            # keep-out: window/niche recesses (unit needs solid wall behind it)
            n_cells += 1
            if len(uv_hole):
                in_hole = ((np.abs(uv_hole[:, 0] - cu) < W / 2 + 5) &
                           (np.abs(uv_hole[:, 1] - cv_) < H / 2 + 5))
                if int(in_hole.sum()) >= 3:
                    n_hole_rej += 1
                    continue
            # clearance to nearest obstacle point
            if len(uv_obst):
                dx = np.maximum(np.abs(uv_obst[:, 0] - cu) - W / 2, 0)
                dy = np.maximum(np.abs(uv_obst[:, 1] - cv_) - H / 2, 0)
                d_obst = np.hypot(dx, dy)
                if int(np.sum(d_obst == 0.0)) >= 3:
                    n_obst_rej += 1
                    continue                                 # obstacle cluster inside rect
                outside = d_obst[d_obst > 0]
                clear = float(outside.min()) if len(outside) else 0.0
            else:
                clear = 1e3
            center_3d = origin + (cu * u_ax + cv_ * v_ax) / cm_per_unit
            dist_ruck = float(np.linalg.norm(center_3d - ruck_X)) * cm_per_unit
            mount_h = cv_ - v_floor
            score = (args.w_dist * dist_ruck
                     + args.w_clear * max(0.0, args.min_clearance_cm - clear)
                     + args.w_height * abs(mount_h - args.preferred_height_cm))
            cands.append(dict(u_cm=round(cu, 1), v_cm=round(cv_, 1),
                              center_model=center_3d.tolist(),
                              dist_to_rucklauf_cm=round(dist_ruck, 1),
                              clearance_cm=round(clear, 1),
                              mount_height_cm=round(mount_h, 1),
                              score=round(score, 1)))
    print(f"Grid cells: {n_cells}  rejected: {n_hole_rej} (window/recess) "
          f"{n_obst_rej} (obstacles)  valid: {len(cands)}")
    if not cands:
        print("FAILED: no valid placement on the marker wall")
        return

    cands.sort(key=lambda c: c["score"])
    top = []
    for c in cands:
        if all(np.hypot(c["u_cm"] - t["u_cm"], c["v_cm"] - t["v_cm"]) > W for t in top):
            top.append(c)
        if len(top) == 3:
            break

    for i, c in enumerate(top, 1):
        print(f"  #{i}: score={c['score']}  d(Rücklauf)={c['dist_to_rucklauf_cm']}cm  "
              f"clearance={c['clearance_cm']}cm  height={c['mount_height_cm']}cm")

    # ── overlay on the registered frame that best sees the #1 recommendation ─
    best_center = np.array(top[0]["center_model"])
    view_img, view_score = None, -1.0
    for im in rec.images.values():
        cam_i = rec.cameras[im.camera_id]
        P_i = np.asarray(im.cam_from_world().matrix())
        Xc = P_i @ np.append(best_center, 1.0)
        if Xc[2] <= 0:
            continue
        xy = np.asarray(cam_i.img_from_cam(Xc[None, :3] / Xc[2])).ravel()
        w, h = cam_i.width, cam_i.height
        if not (0 <= xy[0] < w and 0 <= xy[1] < h):
            continue
        # prefer the unit centered in view and the camera close to it
        centered = 1.0 - (abs(xy[0] - w / 2) / w + abs(xy[1] - h / 2) / h)
        if centered > view_score:
            view_score, view_img = centered, im
    if view_img is None:
        print("Skipping overlay: no frame clearly sees the recommendation")
        view_img = next(iter(rec.images.values()))
    view_name = view_img.name
    cam = rec.cameras[view_img.camera_id]
    P = np.asarray(view_img.cam_from_world().matrix())
    bgr = cv2.imread(str(args.frames_dir / view_name))

    def project(X_model: np.ndarray):
        Xc = P @ np.append(X_model, 1.0)
        if Xc[2] <= 0:
            return None
        xy = np.asarray(cam.img_from_cam(Xc[None, :3] / Xc[2]))
        return tuple(np.round(xy.ravel()[:2]).astype(int))

    colors = [(0, 220, 0), (0, 200, 255), (255, 120, 0)]
    for i, c in enumerate(top):
        corners = []
        for du, dv in [(-W/2, -H/2), (W/2, -H/2), (W/2, H/2), (-W/2, H/2)]:
            X = origin + ((c["u_cm"] + du) * u_ax + (c["v_cm"] + dv) * v_ax) / cm_per_unit
            p = project(X)
            if p is None:
                corners = []
                break
            corners.append(p)
        if corners:
            cv2.polylines(bgr, [np.array(corners)], True, colors[i % 3], 3)
            cv2.putText(bgr, f"#{i+1} d={c['dist_to_rucklauf_cm']:.0f}cm",
                        (corners[0][0], corners[0][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i % 3], 2, cv2.LINE_AA)
    rp = project(ruck_X)
    if rp:
        cv2.drawMarker(bgr, rp, (255, 0, 0), cv2.MARKER_STAR, 26, 3)
        cv2.putText(bgr, "Ruecklauf", (rp[0] + 12, rp[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    out_img = args.sfm_dir / "placement_3d.jpg"
    cv2.imwrite(str(out_img), bgr)

    out = dict(cm_per_unit=cm_per_unit,
               unit_wh_cm=[W, H],
               rucklauf=dict(model_xyz=ruck_X.tolist(), frame=ruck_frame,
                             px=list(ruck_px) if ruck_px else None,
                             confidence=ruck_confidence),
               wall=dict(origin_model=origin.tolist(), u=u_ax.tolist(),
                         v=v_ax.tolist(), normal=n_ax.tolist()),
               floor_z_cm=z_floor,
               candidates_evaluated=len(cands),
               top3=top)
    (args.sfm_dir / "placement_3d.json").write_text(json.dumps(out, indent=2))
    print(f"Saved {args.sfm_dir / 'placement_3d.json'}")
    print(f"Saved {out_img}")


if __name__ == "__main__":
    main()
