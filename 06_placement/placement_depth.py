"""
placement_depth.py — placement recommendation from DENSE monocular depth,
anchored to real SfM camera calibration, instead of the sparse SfM point
cloud placement_3d.py relies on for wall/obstacle evidence.

Why this exists: painted walls carry almost no SIFT-matchable texture, so
COLMAP's sparse reconstruction is structurally sparse ON THE WALL itself —
independent of video quality or coverage (confirmed against current
research: this is a known COLMAP limitation on textureless indoor
surfaces, not a bug in this pipeline). placement_3d.py's --min-wall-inliers
evidence floor correctly refuses many rooms for exactly this reason, and a
first attempt at "densifying" its sparse cloud with monocular depth (still
in placement_3d.py, --no-densify-depth to disable) only patches wall-surface
evidence — the GDINO obstacle/window masks there are STILL lifted through
the same sparse SfM correspondences (lift_mask_to_3d), so windows (also
texture-poor: glass) stay under-evidenced too (confirmed on Renate_Hefele:
forcing denser sampling passed the wall-evidence floor, then still placed
the recommendation on the window, because the window mask itself was
sparse).

This script sidesteps sparse SfM entirely for geometry: it backprojects
EVERY pixel of one frame's dense metric depth map into 3D using that
frame's REAL calibrated intrinsics/pose from the SfM reconstruction (still
needed for accurate focal length, real metric scale, and Rücklauf
localization — none of which were ever the bottleneck), then does wall
fitting, obstacle lifting, AND window/hole detection all from that same
dense cloud. No "not enough native points" failure mode is possible by
construction — a whole image's worth of pixels backprojects into
thousands of points regardless of wall texture.

Trade-off: monocular depth is a single, unverified view (no multi-frame
triangulation cross-check), so its ABSOLUTE geometry is only as good as
the depth model plus a scale-alignment step against whatever sparse SfM
points DO exist in that frame (usually plenty, off the wall — on pipes,
equipment, floor). If too few exist to align confidently, this script
refuses rather than trusting raw monocular depth blindly.

Usage:
  python 06_placement/placement_depth.py output/sfm/IMG_3126 dataset/frames/IMG_3126_sfm
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
sys.path.insert(0, str(_ROOT / "experiments"))
sys.path.insert(0, str(_ROOT / "08_depth"))
sys.path.insert(0, str(_ROOT / "06_placement"))
from room_dims import load_filtered_points, up_from_cameras, rotation_to_z, density_peak
from placement_3d import (INVALID_P3D, locate_rucklauf, load_gdino_sam2, detect_and_segment,
                          wall_frame_ransac_multi, save_segmentation_viz, save_depth_viz)
from depth_models import DEPTH_MODELS


def backproject_frame(cam, P, depth_cm, cm_per_unit, stride, max_depth_cm=800.0):
    """Dense per-pixel backprojection of one frame's metric depth map into
    MODEL coordinates, using the frame's real calibrated intrinsics (via
    pycolmap's own cam_from_img — handles lens distortion correctly) and
    real SfM pose. Returns (Xw [N,3], xs [N], ys [N]) — pixel coords kept
    alongside so GDINO mask lookups can select subsets directly, no second
    lift step needed."""
    R, t = P[:, :3], P[:, 3]
    h, w = depth_cm.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    ys, xs = ys.ravel(), xs.ravel()
    d = depth_cm[ys, xs]
    valid = np.isfinite(d) & (d > 20) & (d < max_depth_cm)
    ys, xs, d = ys[valid], xs[valid], d[valid]
    pix = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    rays = cam.cam_from_img(pix)
    Xc_cm = np.concatenate([rays * d[:, None], d[:, None]], axis=1)
    Xw = ((Xc_cm / cm_per_unit) - t) @ R
    # cam_from_img can return NaN for some pixels at camera-model/distortion
    # edge cases — the `d` finiteness check above doesn't catch a NaN RAY,
    # only a NaN depth. An uncaught NaN point silently poisons every
    # downstream np.percentile() call (NaN propagates, comparisons against
    # NaN are always False so the "too short" sanity check doesn't even
    # catch it), which crashed np.arange() with "cannot compute length" on
    # Bjoern-Harald_Malluche — confirmed real, not a one-off.
    finite = np.all(np.isfinite(Xw), axis=1)
    return Xw[finite], xs[finite], ys[finite]


def dense_mask_xyz(mask, xs, ys, Xw):
    """Subset of the dense backprojected cloud whose source pixel falls
    inside a 2D boolean mask — the dense analogue of placement_3d.py's
    lift_mask_to_3d, without depending on sparse SfM 2D<->3D tracks."""
    if mask is None:
        return np.empty((0, 3))
    h, w = mask.shape[:2]
    inb = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    sel = np.zeros(len(xs), dtype=bool)
    sel[inb] = mask[ys[inb], xs[inb]] > 0
    return Xw[sel]


def masks_union_xyz(masks, xs, ys, Xw):
    """Dense points under the UNION of several masks, each point counted
    once. Concatenating dense_mask_xyz() per mask instead (an earlier
    version of this file did) double/triple-counts any pixel covered by
    more than one overlapping detection — real bug hit multi-prompt-
    ensembled 'wall' detections (3 phrasings, often heavily overlapping
    boxes): the gated pool came out LARGER than the entire dense cloud
    (456627 'wall' points from a 291600-point cloud), and downstream that
    fed a degenerate (NaN) percentile into the grid-search arange call and
    crashed. Union first, backproject once."""
    if not masks:
        return np.empty((0, 3))
    h, w = masks[0].shape[:2]
    inb = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    union_mask = np.zeros((h, w), dtype=bool)
    for m in masks:
        union_mask |= (m > 0)
    sel = np.zeros(len(xs), dtype=bool)
    sel[inb] = union_mask[ys[inb], xs[inb]]
    return Xw[sel]


def main():
    parser = argparse.ArgumentParser(description="Placement recommendation from dense monocular depth")
    parser.add_argument("sfm_dir", type=Path)
    parser.add_argument("frames_dir", type=Path)
    parser.add_argument("--model", default="1")
    parser.add_argument("--unit-wh-cm", type=float, nargs=2, default=[60.0, 40.0])
    parser.add_argument("--min-clearance-cm", type=float, default=10.0)
    parser.add_argument("--preferred-height-cm", type=float, default=120.0)
    parser.add_argument("--min-bottom-cm", type=float, default=20.0)
    parser.add_argument("--w-dist", type=float, default=1.0)
    parser.add_argument("--w-clear", type=float, default=3.0)
    parser.add_argument("--w-height", type=float, default=0.5)
    parser.add_argument("--w-elec", type=float, default=1.0)
    parser.add_argument("--rucklauf-frame", default=None)
    parser.add_argument("--rucklauf-px", type=float, nargs=2, default=None)
    parser.add_argument("--no-rucklauf", action="store_true")
    parser.add_argument("--pipe-prompt", nargs="+", default=["pipe", "metal pipe"])
    parser.add_argument("--fitting-prompt", nargs="+",
                        default=["valve", "trap", "fitting", "gauge", "meter"])
    parser.add_argument("--electrical-prompt", nargs="+",
                        default=["electrical panel", "fuse box", "switch", "outlet", "junction box"])
    parser.add_argument("--window-prompt", default="window")
    parser.add_argument("--wall-prompt", nargs="+", default=["wall", "plaster wall", "concrete wall"],
                        help="GDINO+SAM2 semantic wall mask, used to GATE which dense points are "
                             "even eligible for wall-plane RANSAC — cheap, independent check that "
                             "a candidate plane is only fit from pixels labeled 'wall', instead of "
                             "letting every surface in the room (tanks, ledges, equipment) compete "
                             "on point count alone. Complementary to --min-height-extent-cm, not a "
                             "replacement — a wall mask miss still falls back to the full cloud.")
    parser.add_argument("--no-wall-gate", action="store_true",
                        help="Disable wall-mask gating, search the full dense cloud (old behavior).")
    parser.add_argument("--gdino-threshold", type=float, default=0.3)
    parser.add_argument("--depth-model", choices=list(DEPTH_MODELS), default="metric_anything")
    parser.add_argument("--stride", type=int, default=2,
                        help="Pixel stride for dense backprojection — this is the ENTIRE wall/"
                             "obstacle evidence source here (no sparse SfM cloud fallback), so "
                             "default is much denser than placement_3d.py's optional densifier.")
    parser.add_argument("--min-wall-pts", type=int, default=300,
                        help="Sanity floor on RANSAC wall inliers — deliberately low. A dense "
                             "single-frame cloud has thousands of points on any real wall by "
                             "construction; this only exists to catch a genuinely degenerate fit "
                             "(e.g. depth model failure), not to gate on evidence density the way "
                             "placement_3d.py's --min-wall-inliers has to for sparse SfM points.")
    parser.add_argument("--max-walls-searched", type=int, default=15)
    args = parser.parse_args()

    scale_info = json.loads((args.sfm_dir / "scale.json").read_text())
    cm_per_unit = scale_info["cm_per_unit"]
    rec = pycolmap.Reconstruction(str(args.sfm_dir / "sparse" / args.model))
    print(f"Model: {rec.num_reg_images()} images | scale {cm_per_unit:.3f} cm/unit")

    up = up_from_cameras(rec)
    pts_cm = load_filtered_points(rec) * cm_per_unit
    R_up = rotation_to_z(up)
    z_all = (pts_cm @ R_up.T)[:, 2]
    z_floor = density_peak(z_all, 0.5, 35.0)
    print(f"Floor at z={z_floor:.1f} cm (gravity-aligned)")

    override = None
    if args.rucklauf_frame and args.rucklauf_px:
        override = (args.rucklauf_frame, args.rucklauf_px[0], args.rucklauf_px[1])
    ruck_X = ruck_frame = ruck_px = None
    ruck_confidence = "none"
    if args.no_rucklauf:
        print("RÜCKLAUF: skipped (--no-rucklauf) — free-wall-space mode")
    else:
        try:
            ruck_X, ruck_frame, ruck_px, ruck_confidence = locate_rucklauf(
                rec, args.frames_dir, override, cm_per_unit, None)
            print(f"RÜCKLAUF: found ({ruck_confidence} confidence) — {ruck_frame}")
        except RuntimeError as e:
            if override:
                raise
            print(f"RÜCKLAUF: not found ({e}) — continuing in free-wall-space mode")

    det_name = ruck_frame or max(
        rec.images.values(),
        key=lambda im: sum(1 for p in im.points2D if p.point3D_id != INVALID_P3D)
    ).name
    det_img = next(im for im in rec.images.values() if im.name == det_name)
    cam = rec.cameras[det_img.camera_id]
    P = np.asarray(det_img.cam_from_world().matrix())
    R_det, t_det = P[:, :3], P[:, 3]
    bgr = cv2.imread(str(args.frames_dir / det_name))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    load_fn, predict_fn = DEPTH_MODELS[args.depth_model]
    dm = load_fn(device)
    depth_cm = predict_fn(rgb, dm, device, focal_px=cam.focal_length_x)
    save_depth_viz(depth_cm, args.sfm_dir / "depth_map_dense.png")
    print(f"Saved {args.sfm_dir / 'depth_map_dense.png'} ({args.depth_model}, real focal length "
          f"{cam.focal_length_x:.0f}px from SfM calibration)")

    # Scale-align against whatever sparse SfM points DO exist in this frame
    # (typically plenty — pipes, equipment, floor — even when the WALL
    # itself has almost none). Refuses rather than trusting raw monocular
    # depth un-anchored, same principle as placement_3d.py's densifier.
    sfm_z, mono_z = [], []
    for p2d in det_img.points2D:
        pid = p2d.point3D_id
        if pid == INVALID_P3D or pid not in rec.points3D:
            continue
        Xc = R_det @ rec.points3D[pid].xyz + t_det
        if Xc[2] <= 0:
            continue
        px, py = int(round(p2d.xy[0])), int(round(p2d.xy[1]))
        if not (0 <= py < depth_cm.shape[0] and 0 <= px < depth_cm.shape[1]):
            continue
        d = depth_cm[py, px]
        if not np.isfinite(d) or d <= 0:
            continue
        sfm_z.append(Xc[2] * cm_per_unit)
        mono_z.append(d)
    if len(sfm_z) < 15:
        print(f"FAILED: only {len(sfm_z)} co-visible SfM points in {det_name} — too few to "
              f"trust a scale alignment for monocular depth. Try --rucklauf-frame to pick a "
              f"different, better-covered frame.")
        return
    ratios = np.array(sfm_z) / np.array(mono_z)
    med = float(np.median(ratios))
    spread = float(np.median(np.abs(ratios - med)) / max(med, 1e-6))
    print(f"Scale alignment: {len(sfm_z)} co-visible points, ratio {med:.2f} (spread {spread * 100:.0f}%)")
    if spread >= 0.25:
        print(f"FAILED: scale ratio too inconsistent (spread {spread * 100:.0f}%) — monocular "
              f"depth doesn't agree with SfM closely enough on this frame to trust it.")
        return
    depth_cm = depth_cm * med

    Xw, xs, ys = backproject_frame(cam, P, depth_cm, cm_per_unit, args.stride)
    print(f"Dense backprojection: {len(Xw)} points from {det_name} (stride {args.stride})")

    print(f"GDINO+SAM2 obstacle detection on {det_name}:")
    pipe_boxes, pipe_masks = detect_and_segment(rgb, device, args.pipe_prompt, args.gdino_threshold)
    pipe_xyz = masks_union_xyz(pipe_masks, xs, ys, Xw)
    print(f"  pipes: {len(pipe_masks)} detection(s) -> {len(pipe_xyz)} dense 3D points")
    fit_boxes, fit_masks = detect_and_segment(rgb, device, args.fitting_prompt, args.gdino_threshold)
    fit_xyz = masks_union_xyz(fit_masks, xs, ys, Xw)
    print(f"  fittings: {len(fit_masks)} detection(s) -> {len(fit_xyz)} dense 3D points")
    elec_boxes, elec_masks = detect_and_segment(rgb, device, args.electrical_prompt, args.gdino_threshold)
    elec_xyz = masks_union_xyz(elec_masks, xs, ys, Xw)
    print(f"  electricals: {len(elec_masks)} detection(s) -> {len(elec_xyz)} dense 3D points")
    window_boxes, window_masks = detect_and_segment(rgb, device, args.window_prompt, args.gdino_threshold)
    window_box_viz, window_mask_viz = None, None
    window_xyz = np.empty((0, 3))
    if window_masks:
        img_area = rgb.shape[0] * rgb.shape[1]
        plausible_idx = [i for i, m in enumerate(window_masks) if m.astype(bool).sum() < 0.5 * img_area]
        plausible = [window_masks[i] for i in plausible_idx]
        window_xyz = masks_union_xyz(plausible, xs, ys, Xw)
        print(f"  window: {len(plausible)}/{len(window_masks)} plausible -> {len(window_xyz)} dense 3D points")
        if plausible_idx:
            areas = [(window_boxes[i][2] - window_boxes[i][0]) * (window_boxes[i][3] - window_boxes[i][1])
                    for i in plausible_idx]
            best_i = plausible_idx[int(np.argmax(areas))]
            window_box_viz, window_mask_viz = window_boxes[best_i], window_masks[best_i]

    save_segmentation_viz(bgr, args.sfm_dir / "gdino_segmentation_dense.png",
                          elec_boxes=elec_boxes, elec_masks=elec_masks,
                          window_box=window_box_viz, window_mask=window_mask_viz,
                          pipe_boxes=pipe_boxes, pipe_masks=pipe_masks,
                          fitting_boxes=fit_boxes, fitting_masks=fit_masks,
                          ruck_px=ruck_px)
    print(f"Saved {args.sfm_dir / 'gdino_segmentation_dense.png'}")

    # Wall-plane candidate GATING: restrict which dense points are even
    # eligible for RANSAC to pixels GDINO+SAM2 semantically labels "wall" —
    # cheap, independent of point density/depth accuracy, so tanks/ledges/
    # equipment can't out-compete the real wall on point count alone.
    # Falls back to the full cloud if the wall mask is empty or too small
    # to be a real gate (a miss here shouldn't make the script MORE broken
    # than not gating at all).
    Xw_for_ransac = Xw
    if not args.no_wall_gate:
        wall_boxes, wall_masks = detect_and_segment(rgb, device, args.wall_prompt, args.gdino_threshold)
        wall_xyz = masks_union_xyz(wall_masks, xs, ys, Xw)
        print(f"  wall: {len(wall_masks)} detection(s) -> {len(wall_xyz)} dense 3D points")
        if len(wall_xyz) >= 500:
            Xw_for_ransac = wall_xyz
            print(f"Wall-gated RANSAC pool: {len(Xw_for_ransac)} points (of {len(Xw)} total)")
        else:
            print(f"Wall mask too small ({len(wall_xyz)} pts) to gate on — searching the full cloud")

    ref_point = ruck_X if ruck_X is not None else Xw.mean(axis=0)
    near_cm = 250.0 if ruck_X is not None else 1e6
    candidates_wf = wall_frame_ransac_multi(Xw_for_ransac, up, ref_point, z_floor, cm_per_unit,
                                            near_cm=near_cm, min_height_extent_cm=130.0)
    if not candidates_wf:
        print("FAILED: no dominant vertical wall found in the dense cloud")
        return
    print(f"Wall candidates found: {len(candidates_wf)}")
    by_inliers = sorted(candidates_wf, key=lambda c: -c["n_inl"])[:args.max_walls_searched]
    by_angle = sorted(candidates_wf, key=lambda c: c["view_angle"])[:args.max_walls_searched]
    seen, search_set = set(), []
    for c in by_angle + by_inliers:
        if id(c) not in seen:
            seen.add(id(c))
            search_set.append(c)

    W, H = args.unit_wh_cm
    cams = np.array([det_img.projection_center()])

    def search_wall(origin, u_ax, v_ax, n_ax, n_inl, wall_idx):
        d_plane = (Xw - origin) @ n_ax * cm_per_unit
        uv = np.stack([(Xw - origin) @ u_ax, (Xw - origin) @ v_ax], axis=1) * cm_per_unit
        front_sign = np.sign(float((cams[0] - origin) @ n_ax))
        d_front = d_plane * front_sign

        uv_wall = uv[np.abs(d_plane) < 8.0]
        uv_obst = uv[(d_front >= 8.0) & (d_front < 40.0)]
        uv_hole = uv[(d_front <= -8.0) & (d_front > -80.0)]

        def to_uv(xyz):
            if len(xyz) == 0:
                return np.empty((0, 2))
            return np.stack([(xyz - origin) @ u_ax, (xyz - origin) @ v_ax], axis=1) * cm_per_unit

        elec_uv = to_uv(elec_xyz)
        for xyz in (pipe_xyz, fit_xyz, elec_xyz):
            extra = to_uv(xyz)
            if len(extra):
                uv_obst = np.concatenate([uv_obst, extra])
        uv_win = to_uv(window_xyz)
        if len(uv_win):
            uv_hole = np.concatenate([uv_hole, uv_win]) if len(uv_hole) else uv_win

        if len(uv_wall) < 200:
            print(f"  wall[{wall_idx}]: not enough dense wall coverage ({len(uv_wall)} pts), skipping")
            return []
        uv_evidence = np.concatenate([uv_wall, uv_obst] + ([uv_hole] if len(uv_hole) else []))
        lo = np.percentile(uv_evidence, 1, axis=0)
        hi = np.percentile(uv_evidence, 99, axis=0)

        origin_z = float(((origin * cm_per_unit) @ R_up.T)[2])
        v_floor = z_floor - origin_z
        # Secondary backstop, cheap: wall_frame_ransac_multi's own
        # min_height_extent_cm already screens candidates on THEIR inlier
        # extent before this function is even called (root-caused on
        # Renate_Hefele: NOT short foreground objects out-competing the
        # real wall — the whole dense cloud from that frame capped at
        # 64cm above floor regardless of candidate, a depth-model accuracy
        # limit, not a candidate-selection bug — see placement_depth.py's
        # module docstring / context.txt session log). This check instead
        # catches the case where the WALL itself passed but the combined
        # wall+obstacle+hole evidence (uv_evidence, not just wall inliers)
        # still doesn't reach a usable mounting height.
        if hi[1] < v_floor + 100.0:
            print(f"  wall[{wall_idx}]: evidence only reaches {hi[1] - v_floor:.0f}cm above floor "
                  f"— too short to search a 120cm-preferred mount height, skipping")
            return []
        ruck_uv = None
        if ruck_X is not None:
            ruck_uv = np.array([float((ruck_X - origin) @ u_ax), float((ruck_X - origin) @ v_ax)]) * cm_per_unit

        step = 5.0
        cands_out = []
        n_cells = n_hole_rej = n_obst_rej = 0
        for cu in np.arange(lo[0] + W / 2, hi[0] - W / 2 + 1e-6, step):
            for cv_ in np.arange(max(lo[1] + H / 2, v_floor + args.min_bottom_cm + H / 2),
                                 hi[1] - H / 2 + 1e-6, step):
                n_cells += 1
                if len(uv_hole):
                    in_hole = ((np.abs(uv_hole[:, 0] - cu) < W / 2 + 5) &
                               (np.abs(uv_hole[:, 1] - cv_) < H / 2 + 5))
                    if int(in_hole.sum()) >= 8:
                        n_hole_rej += 1
                        continue
                if len(uv_obst):
                    dx = np.maximum(np.abs(uv_obst[:, 0] - cu) - W / 2, 0)
                    dy = np.maximum(np.abs(uv_obst[:, 1] - cv_) - H / 2, 0)
                    d_obst = np.hypot(dx, dy)
                    if int(np.sum(d_obst == 0.0)) >= 8:
                        n_obst_rej += 1
                        continue
                    outside = d_obst[d_obst > 0]
                    clear = float(outside.min()) if len(outside) else 0.0
                else:
                    clear = 1e3
                center_3d = origin + (cu * u_ax + cv_ * v_ax) / cm_per_unit
                dist_ruck = float(np.linalg.norm(center_3d - ruck_X)) * cm_per_unit if ruck_X is not None else None
                dist_elec = float(np.min(np.linalg.norm(elec_uv - [cu, cv_], axis=1))) if len(elec_uv) else 0.0
                mount_h = cv_ - v_floor
                score = (args.w_elec * dist_elec
                         + args.w_clear * max(0.0, args.min_clearance_cm - clear)
                         + args.w_height * abs(mount_h - args.preferred_height_cm))
                if dist_ruck is not None:
                    score += args.w_dist * dist_ruck
                cands_out.append(dict(u_cm=round(float(cu), 1), v_cm=round(float(cv_), 1),
                                      center_model=center_3d.tolist(),
                                      dist_to_rucklauf_cm=round(dist_ruck, 1) if dist_ruck is not None else None,
                                      dist_to_electrical_cm=round(dist_elec, 1),
                                      clearance_cm=round(clear, 1),
                                      mount_height_cm=round(mount_h, 1),
                                      score=round(score, 1),
                                      wall_idx=wall_idx, wall_n_inl=n_inl,
                                      wall_origin=origin.tolist(), wall_u=u_ax.tolist(),
                                      wall_v=v_ax.tolist(), wall_n=n_ax.tolist()))
        print(f"  wall[{wall_idx}] ({n_inl} inl): {n_cells} cells, {n_hole_rej} window-rejected, "
              f"{n_obst_rej} obstacle-rejected, {len(cands_out)} valid (wall-pts={len(uv_wall)})")
        return cands_out

    all_cands = []
    for wall_idx, wall in enumerate(search_set):
        wall_tilt = float(np.degrees(np.arcsin(abs(wall["n"] @ up))))
        if wall_tilt > 25.0 or wall["view_angle"] > 60.0:
            continue
        all_cands.extend(search_wall(wall["origin"], wall["u"], wall["v"], wall["n"],
                                     wall["n_inl"], wall_idx))

    trustworthy = [c for c in all_cands if c["wall_n_inl"] >= args.min_wall_pts]
    if not trustworthy:
        print(f"FAILED: {len(all_cands)} candidate(s) found, but none on a wall with "
              f"--min-wall-pts {args.min_wall_pts} inliers — even the dense cloud couldn't "
              f"support a wall here (frame likely doesn't clearly show one flat surface).")
        return
    cands = sorted(trustworthy, key=lambda c: c["score"])
    top = []
    for c in cands:
        if all(c["wall_idx"] != t["wall_idx"] or
               np.hypot(c["u_cm"] - t["u_cm"], c["v_cm"] - t["v_cm"]) > W
               for t in top):
            top.append(c)
        if len(top) == 3:
            break

    for i, c in enumerate(top, 1):
        dstr = f"d(Rücklauf)={c['dist_to_rucklauf_cm']}cm  " if c.get("dist_to_rucklauf_cm") is not None else ""
        print(f"  #{i}: wall[{c['wall_idx']}] score={c['score']}  {dstr}"
              f"d(electrical)={c.get('dist_to_electrical_cm', 0)}cm  "
              f"clearance={c['clearance_cm']}cm  height={c['mount_height_cm']}cm")

    origin = np.array(top[0]["wall_origin"])
    u_ax = np.array(top[0]["wall_u"])
    v_ax = np.array(top[0]["wall_v"])

    def project(X_model):
        Xc = P @ np.append(X_model, 1.0)
        if Xc[2] <= 0:
            return None
        xy = np.asarray(cam.img_from_cam(Xc[None, :3] / Xc[2]))
        return tuple(np.round(xy.ravel()[:2]).astype(int))

    out_bgr = bgr.copy()
    colors = [(0, 220, 0), (0, 200, 255), (255, 120, 0)]
    for i, c in enumerate(top):
        # Same fix as placement_3d.py: each candidate's (u_cm, v_cm) is in
        # ITS OWN wall's frame, not necessarily top[0]'s — using the global
        # origin/u_ax/v_ax for every candidate corrupted #2/#3 whenever they
        # came from a different wall.
        c_origin = np.array(c["wall_origin"])
        c_u = np.array(c["wall_u"])
        c_v = np.array(c["wall_v"])
        corners = []
        for du, dv in [(-W / 2, -H / 2), (W / 2, -H / 2), (W / 2, H / 2), (-W / 2, H / 2)]:
            X = c_origin + ((c["u_cm"] + du) * c_u + (c["v_cm"] + dv) * c_v) / cm_per_unit
            p = project(X)
            if p is None:
                corners = []
                break
            corners.append(p)
        if corners:
            cv2.polylines(out_bgr, [np.array(corners)], True, colors[i % 3], 3)
            label = (f"#{i+1} d={c['dist_to_rucklauf_cm']:.0f}cm" if c.get("dist_to_rucklauf_cm") is not None
                     else f"#{i+1} clr={c['clearance_cm']:.0f}cm")
            cv2.putText(out_bgr, label, (corners[0][0], corners[0][1] - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i % 3], 2, cv2.LINE_AA)
    if ruck_X is not None:
        rp = project(ruck_X)
        if rp:
            cv2.drawMarker(out_bgr, rp, (255, 0, 0), cv2.MARKER_STAR, 26, 3)
            cv2.putText(out_bgr, "Ruecklauf", (rp[0] + 12, rp[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    out_img = args.sfm_dir / "placement_depth.jpg"
    cv2.imwrite(str(out_img), out_bgr)

    out = dict(cm_per_unit=cm_per_unit, unit_wh_cm=[W, H], detection_frame=det_name,
               scale_alignment=dict(n_points=len(sfm_z), ratio=med, spread_pct=round(spread * 100, 1)),
               rucklauf=dict(model_xyz=ruck_X.tolist() if ruck_X is not None else None, frame=ruck_frame,
                             px=list(ruck_px) if ruck_px else None, confidence=ruck_confidence),
               wall=dict(origin_model=origin.tolist(), u=u_ax.tolist(), v=v_ax.tolist()),
               floor_z_cm=z_floor, candidates_evaluated=len(cands), top3=top)
    (args.sfm_dir / "placement_depth.json").write_text(json.dumps(out, indent=2))
    print(f"Saved {args.sfm_dir / 'placement_depth.json'}")
    print(f"Saved {out_img}")


if __name__ == "__main__":
    main()
