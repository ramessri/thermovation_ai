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
sys.path.insert(0, str(_ROOT / "experiments"))
from room_dims import load_filtered_points, up_from_cameras, rotation_to_z, density_peak

INVALID_P3D = 2**63 - 1


# ── GDINO+SAM2 obstacle detection (pipes/fittings/electricals/window) ───────
# Ported from 06_placement/single_image_placement.py's validated version:
# real electrical-panel and small-fitting (valve/trap/gauge) misses there
# caused actual bad placements (see that script's changelog comments) before
# being caught by folding their masks into the obstacle set the same way
# pipes already were. Here the fusion is via lift_mask_to_3d (real SfM
# triangulation across registered frames), not a monocular depth guess —
# strictly more reliable than the single-image version, not a port-for-
# parity exercise.

def lift_mask_to_3d(rec: pycolmap.Reconstruction, img, mask: np.ndarray) -> list[int]:
    """point3D_ids whose 2D observation in this image falls inside the mask."""
    ids = []
    h, w = mask.shape[:2]
    for p2d in img.points2D:
        pid = p2d.point3D_id
        if pid == INVALID_P3D or pid not in rec.points3D:
            continue
        x, y = int(round(p2d.xy[0])), int(round(p2d.xy[1]))
        if 0 <= y < h and 0 <= x < w and mask[y, x] > 0:
            ids.append(pid)
    return ids


_GDINO_SAM2_CACHE = {}


def load_gdino_sam2(device: str):
    if "models" not in _GDINO_SAM2_CACHE:
        from experiment_pipeline import load_gdino, load_sam2
        gdino_proc, gdino_model = load_gdino(device)
        sam2 = load_sam2(device)
        _GDINO_SAM2_CACHE["models"] = (gdino_proc, gdino_model, sam2)
    return _GDINO_SAM2_CACHE["models"]


def detect_and_segment(image_rgb, device, prompts, threshold: float):
    """GDINO boxes -> SAM2 masks. `prompts` may be a single phrase or a list
    to ensemble (union, no dedup needed here — redundant obstacle points
    from overlapping detections are harmless, unlike in the ranking-sensitive
    single-image script)."""
    from experiment_pipeline import detect_gdino, sam2_from_boxes
    gdino_proc, gdino_model, sam2 = load_gdino_sam2(device)
    prompts = [prompts] if isinstance(prompts, str) else list(prompts)
    all_boxes = []
    for p in prompts:
        boxes, labels, scores = detect_gdino(image_rgb, gdino_proc, gdino_model, p, threshold, device)
        if len(boxes):
            all_boxes.append(boxes)
    boxes = np.concatenate(all_boxes, axis=0) if all_boxes else np.empty((0, 4))
    masks = sam2_from_boxes(sam2, image_rgb, boxes) if len(boxes) else []
    return boxes, masks


# ── room-layout wall segmentation (ADE20K semantic seg) ─────────────────────
# A purpose-built alternative to the "wall" GDINO+SAM2 prompt: a real
# semantic segmentation model (per-pixel, not box-then-mask) trained on
# ADE20K's ~20k diverse scene images, which include "wall" as a first-class
# category (id 0) alongside floor/ceiling/window/door. Considered ST-RoomNet
# (the literal room-LAYOUT tool — cuboid corner/plane estimation) first, but
# its LSUN training data is clean box-shaped bedrooms; a cluttered, often
# irregular boiler room with equipment occluding most walls is exactly the
# kind of scene that assumption breaks on. ADE20K segmentation makes a
# weaker claim (just "which pixels are wall," no cuboid-fitting) that's a
# better match to what's actually needed here, and ADE20K's scene diversity
# gives it a better shot at generalizing to a utility room than a
# bedroom-specific layout dataset — still needs the same skeptical
# per-video visual verification as everything else in this pipeline.

_WALL_SEG_CACHE = {}


def load_wall_seg_model(device: str, model_id: str = "nvidia/segformer-b2-finetuned-ade-512-512"):
    if model_id not in _WALL_SEG_CACHE:
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForSemanticSegmentation.from_pretrained(model_id).to(device).eval()
        wall_id = next(i for i, l in model.config.id2label.items() if l.lower() == "wall")
        print(f"  ADE20K wall segmentation loaded ({model_id}, wall=class {wall_id})")
        _WALL_SEG_CACHE[model_id] = (processor, model, wall_id)
    return _WALL_SEG_CACHE[model_id]


def segment_wall_mask(image_rgb: np.ndarray, device: str,
                      model_id: str = "nvidia/segformer-b2-finetuned-ade-512-512") -> np.ndarray:
    """Dense boolean wall mask at the input image's resolution."""
    import torch
    from PIL import Image as PILImage
    processor, model, wall_id = load_wall_seg_model(device, model_id)
    pil_img = PILImage.fromarray(image_rgb)
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    seg = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[(image_rgb.shape[0], image_rgb.shape[1])])[0]
    return (seg == wall_id).cpu().numpy()


def masks_to_xyz(rec, img, masks):
    """Lift 2D masks -> real triangulated 3D points (via lift_mask_to_3d),
    in MODEL units (not yet projected onto any particular wall — a
    candidate wall isn't chosen until wall_frame_ransac_multi runs, and
    projecting has to happen per-wall, once for each candidate evaluated)."""
    if not masks:
        return np.empty((0, 3))
    pids = set()
    for m in masks:
        pids.update(lift_mask_to_3d(rec, img, m))
    if not pids:
        return np.empty((0, 3))
    return np.array([rec.points3D[pid].xyz for pid in pids])


def xyz_to_uv(xyz, origin, u_ax, v_ax, cm_per_unit):
    if len(xyz) == 0:
        return np.empty((0, 2))
    return np.stack([(xyz - origin) @ u_ax, (xyz - origin) @ v_ax], axis=1) * cm_per_unit


def wall_frame_ransac_multi(pts_model, up_world, ref_model, z_floor_cm, cm_per_unit,
                            iters: int = 6000, thresh_cm: float = 6.0, near_cm: float = 250.0,
                            min_inlier_frac: float = 0.05, min_height_extent_cm: float = 100.0):
    """Ported from 06_placement/single_image_placement.py — same 2D
    line-RANSAC wall_frame_ransac() above uses, but tracks every distinct
    well-supported candidate line (deduped by actual line geometry: angle +
    perpendicular offset, not by inlier-set overlap, which fails on one
    long wall — different random samples along its length each explain a
    different sub-region with LOW pairwise overlap despite being the same
    physical surface) instead of keeping only the single best-inlier one.

    Why this matters here even though this script has real SfM triangulation
    (unlike the single-image version): "most inliers" still isn't a
    reliable proxy for "the wall near the Rücklauf" when a room has more
    than one real vertical surface nearby and one of them (e.g. the far
    wall glimpsed through a doorway) simply has more feature points. Real
    failure this was built to catch: a 4358-inlier RANSAC wall on
    Klaus_Rombergg produced a degenerate placement rectangle sitting
    directly on a boiler tank, not a wall at all — a lower-inlier but
    better-conditioned candidate elsewhere would have been fine, but the
    single-candidate version never got to try it.

    Returns a list of dicts (each: origin, u, v, n, n_inl, view_angle,
    height_extent_cm), sorted most-face-on first, or None if nothing
    reaches the minimum inlier count at all.

    min_height_extent_cm rejects a candidate whose own inlier points don't
    reach at least this high above the floor — a real room wall spans
    floor-to-ceiling; a candidate whose evidence tops out at, say, 40cm is
    a short foreground object (a ledge, a tank's flat side), not the wall,
    however many inliers it racked up. Matters most for dense per-pixel
    point clouds (placement_depth.py), where every surface in the room
    gets equal point density and a short object can otherwise out-compete
    the real wall purely on point count — sparse SfM rarely has enough
    points on such objects for this to matter, but the check is cheap and
    correct there too.
    """
    R = rotation_to_z(up_world)
    q = pts_model @ R.T
    zc = q[:, 2] * cm_per_unit
    band = (zc > z_floor_cm + 15) & (zc < z_floor_cm + 230)
    qb = q[band]
    if len(qb) < 50:
        return None
    ref_q = ref_model @ R.T
    d_top = np.hypot((qb[:, 0] - ref_q[0]) * cm_per_unit, (qb[:, 1] - ref_q[1]) * cm_per_unit)
    pool = qb[d_top < near_cm]
    if len(pool) < 50:
        pool = qb
    xy = pool[:, :2]
    t = thresh_cm / cm_per_unit
    rng = np.random.default_rng(0)
    N = len(xy)

    found = []  # list of [inl_count, inl_mask, angle_deg, offset_cm]
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
        if c < 30:
            continue
        angle = float(np.degrees(np.arctan2(e[1], e[0]))) % 180.0
        offset = float(xy[i] @ nrm) * cm_per_unit
        dup_idx = None
        for k, (c2, inl2, a2, o2) in enumerate(found):
            dangle = min(abs(angle - a2), 180 - abs(angle - a2))
            if dangle < 8.0 and abs(offset - o2) < 25.0:
                dup_idx = k
                break
        if dup_idx is None:
            found.append([c, inl, angle, offset])
        elif c > found[dup_idx][0]:
            found[dup_idx] = [c, inl, angle, offset]

    if not found:
        return None
    best_overall = max(c for c, _, _, _ in found)
    candidates = []
    for c, inl, _angle, _offset in found:
        if c < max(30, min_inlier_frac * best_overall):
            continue
        zin_cm = pool[inl, 2] * cm_per_unit
        height_extent_cm = float(zin_cm.max() - z_floor_cm)
        if height_extent_cm < min_height_extent_cm:
            continue
        ptsin = xy[inl]
        c0 = ptsin.mean(0)
        _, _, vt = np.linalg.svd(ptsin - c0, full_matrices=False)
        e2 = vt[0]
        nrm = np.array([-e2[1], e2[0]])
        uq = np.array([e2[0], e2[1], 0.0])
        vq = np.array([0.0, 0.0, 1.0])
        nq = np.array([nrm[0], nrm[1], 0.0])
        oq = np.array([c0[0], c0[1], z_floor_cm / cm_per_unit])
        origin = R.T @ oq
        u = R.T @ uq
        v = R.T @ vq
        n = R.T @ nq
        view_angle = float(np.degrees(np.arccos(np.clip(abs(n[2]), 0, 1))))
        candidates.append(dict(origin=origin, u=u, v=v, n=n, n_inl=c, view_angle=view_angle,
                               height_extent_cm=height_extent_cm))

    candidates.sort(key=lambda cd: cd["view_angle"])
    return candidates if candidates else None


# ── diagnostic visualizations (depth map, GDINO+SAM2 segmentation) ─────────
# Ported from single_image_placement.py — copied rather than imported to
# avoid a circular import (that script already imports blue_circle_candidates
# from this one). The depth map here is purely a visual diagnostic on the
# detection frame; unlike single_image_placement.py, it never feeds the
# actual geometry — this script's placement math runs entirely on real SfM
# triangulation, which is the more reliable signal.

def save_wall_seg_viz(bgr: np.ndarray, wall_mask: np.ndarray, out_path: Path):
    """Green tint over ADE20K's 'wall' class, for visual review alongside
    the GDINO obstacle segmentation and depth map."""
    overlay = bgr.copy()
    overlay[wall_mask] = (0, 220, 0)
    vis = cv2.addWeighted(overlay, 0.4, bgr, 0.6, 0)
    cv2.imwrite(str(out_path), vis)


def save_depth_viz(depth_cm: np.ndarray, out_path: Path):
    valid = np.isfinite(depth_cm)
    lo, hi = np.percentile(depth_cm[valid], [1, 99])
    norm = np.clip((depth_cm - lo) / max(hi - lo, 1e-6), 0, 1)
    norm8 = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm8, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    cv2.imwrite(str(out_path), colored)


def save_segmentation_viz(bgr: np.ndarray, out_path: Path,
                          elec_boxes=None, elec_masks=None,
                          window_box=None, window_mask=None,
                          pipe_boxes=None, pipe_masks=None,
                          fitting_boxes=None, fitting_masks=None,
                          ruck_px=None):
    vis = bgr.copy()
    overlay = bgr.copy()
    if pipe_masks:
        for mask in pipe_masks:
            overlay[mask.astype(bool)] = (255, 200, 0)   # cyan = pipe
    if fitting_masks:
        for mask in fitting_masks:
            overlay[mask.astype(bool)] = (255, 0, 200)   # magenta = fitting
    if elec_masks:
        for mask in elec_masks:
            overlay[mask.astype(bool)] = (0, 200, 255)   # orange = electrical
    if window_mask is not None:
        overlay[window_mask.astype(bool)] = (0, 0, 255)  # red = window
    vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)

    if pipe_boxes is not None:
        for b in pipe_boxes:
            x0, y0, x1, y1 = map(int, b)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 200, 0), 2)
            cv2.putText(vis, "pipe", (x0, max(y0 - 6, 12)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
    if fitting_boxes is not None:
        for b in fitting_boxes:
            x0, y0, x1, y1 = map(int, b)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 0, 200), 2)
            cv2.putText(vis, "fitting", (x0, max(y0 - 6, 12)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 200), 2)
    if elec_boxes is not None:
        for b in elec_boxes:
            x0, y0, x1, y1 = map(int, b)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 200, 255), 2)
            cv2.putText(vis, "electrical", (x0, max(y0 - 6, 12)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
    if window_box is not None:
        x0, y0, x1, y1 = map(int, window_box)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(vis, "window", (x0, max(y0 - 6, 12)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    if ruck_px is not None:
        cv2.circle(vis, (int(ruck_px[0]), int(ruck_px[1])), 8, (255, 0, 0), -1)
        cv2.putText(vis, "Rucklauf", (int(ruck_px[0]) + 10, int(ruck_px[1])),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imwrite(str(out_path), vis)


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
    parser.add_argument("--no-rucklauf", action="store_true",
                        help="Skip Rücklauf localization entirely and report free wall "
                             "space only (clearance/electrical/height scoring, no "
                             "distance-to-Rücklauf term). Also the automatic fallback "
                             "when no blue-cap cue or pipe-pairing signal is found.")
    parser.add_argument("--w-elec", type=float, default=1.0)
    parser.add_argument("--pipe-prompt", nargs="+", default=["pipe", "metal pipe"])
    parser.add_argument("--fitting-prompt", nargs="+",
                        default=["valve", "trap", "fitting", "gauge", "meter"])
    parser.add_argument("--electrical-prompt", nargs="+",
                        default=["electrical panel", "fuse box", "switch", "outlet", "junction box"])
    parser.add_argument("--window-prompt", default="window")
    parser.add_argument("--gdino-threshold", type=float, default=0.3)
    parser.add_argument("--no-gdino", action="store_true",
                        help="Skip GDINO+SAM2 obstacle detection, use only the SfM sparse "
                             "point cloud (original behavior)")
    parser.add_argument("--max-walls-searched", type=int, default=15,
                        help="Cap on distinct RANSAC wall candidates given a full obstacle/grid "
                             "search — bounds runtime. Union of top-N by inlier count and top-N "
                             "by face-on angle, so neither a well-evidenced wall nor a smaller "
                             "well-oriented one loses its slot to the other.")
    parser.add_argument("--min-wall-inliers", type=int, default=2000,
                        help="Minimum RANSAC inlier count for a wall's candidates to be trusted "
                             "in ranking. Real failure this fixed: a 4358-inlier wall on "
                             "Klaus_Rombergg produced a degenerate placement directly on a boiler "
                             "tank — a good angle doesn't mean good evidence for 5cm-grid obstacle "
                             "math. Ignored for the triangulated-marker path (always trustworthy).")
    parser.add_argument("--depth-model", choices=["none", "metric_anything", "depthpro"], default="depthpro",
                        help="Monocular depth model for the detection frame. Used for the "
                             "depth_map.png diagnostic AND (unless --no-densify-depth) to "
                             "densify the wall/obstacle point cloud — see --no-densify-depth.")
    parser.add_argument("--densify-stride", type=int, default=6,
                        help="Pixel stride when backprojecting the depth map for densification — "
                             "lower is denser (more wall-evidence points, slower).")
    parser.add_argument("--no-densify-depth", action="store_true",
                        help="Don't backproject the monocular depth map into the point cloud. "
                             "Root cause this exists to address: plain painted walls have almost "
                             "no SIFT-matchable texture, so COLMAP's sparse reconstruction is "
                             "structurally sparse ON THE WALL regardless of video quality — most "
                             "of the 24-video batch's --min-wall-inliers refusals were this, not "
                             "a placement bug. Densification backprojects the depth map through "
                             "the detection frame's REAL SfM pose/intrinsics (pycolmap's own "
                             "cam_from_img, so real lens distortion is handled), after aligning "
                             "its scale to the SfM's own triangulated points at co-visible pixels "
                             "(median ratio) — raw monocular depth is not trusted un-anchored. "
                             "If fewer than 15 co-visible points exist, or their scale ratio is "
                             "too inconsistent (>25% spread), densification is skipped for that "
                             "video and it falls back to sparse-only behavior.")
    parser.add_argument("--no-wall-seg", action="store_true",
                        help="Skip ADE20K wall segmentation entirely (no wall_segmentation.png "
                             "diagnostic, no lifted points computed at all).")
    parser.add_argument("--wall-seg-hard-gate", action="store_true",
                        help="Use the ADE20K wall mask to HARD-RESTRICT the RANSAC candidate pool "
                             "to only mask-lifted sparse points, instead of just visualizing it. "
                             "Off by default — regressed Klaus_Rombergg (3252 native inliers on the "
                             "correct wall dropped under 500 once non-mask-labeled points were "
                             "excluded, failing the trust floor outright) even with no densify "
                             "involved. The segmentation itself still runs and saves either way.")
    parser.add_argument("--wall-seg-model", default="nvidia/segformer-b2-finetuned-ade-512-512")
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
    if args.no_rucklauf:
        ruck_X = ruck_frame = ruck_px = None
        ruck_confidence = "none"
        print("RÜCKLAUF: skipped (--no-rucklauf) — free-wall-space mode")
    else:
        try:
            ruck_X, ruck_frame, ruck_px, ruck_confidence = locate_rucklauf(
                rec, args.frames_dir, override, cm_per_unit, pipe_candidate)
            where = f" — {ruck_frame} ({ruck_px[0]:.0f},{ruck_px[1]:.0f})" if ruck_frame else ""
            print(f"RÜCKLAUF: found ({ruck_confidence} confidence){where}")
        except RuntimeError as e:
            if override:
                raise
            ruck_X = ruck_frame = ruck_px = None
            ruck_confidence = "none"
            print(f"RÜCKLAUF: not found ({e}) — continuing in free-wall-space mode")

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

    # ── GDINO+SAM2 obstacle detection — runs ONCE, independent of which wall
    # ends up chosen (a candidate wall isn't picked until below). Fuses masks
    # to 3D via the SAME real SfM triangulation the rest of this script
    # already trusts — no monocular depth guessing involved. Electrical
    # boxes are folded in as hard obstacles too, not just scored by
    # proximity — a real failure in the single-image version put a
    # candidate directly on an electrical panel because "closer is better"
    # scoring alone doesn't stop at the edge. ────────────────────────────
    pipe_xyz = fit_xyz = elec_xyz = window_xyz = dense_pts_model = wall_seg_xyz = np.empty((0, 3))
    det_frame_name = None
    if not args.no_gdino:
        det_frame_name = ruck_frame or max(
            rec.images.values(),
            key=lambda im: sum(1 for p in im.points2D if p.point3D_id != INVALID_P3D)
        ).name
        det_img = next((im for im in rec.images.values() if im.name == det_frame_name), None)
        det_bgr = cv2.imread(str(args.frames_dir / det_frame_name)) if det_img else None
        if det_img is not None and det_bgr is not None:
            det_rgb = cv2.cvtColor(det_bgr, cv2.COLOR_BGR2RGB)
            print(f"GDINO+SAM2 obstacle detection on {det_frame_name}:")
            device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            pipe_boxes, pipe_masks = detect_and_segment(det_rgb, device, args.pipe_prompt, args.gdino_threshold)
            pipe_xyz = masks_to_xyz(rec, det_img, pipe_masks)
            print(f"  pipes: {len(pipe_masks)} detection(s) -> {len(pipe_xyz)} 3D points")
            fit_boxes, fit_masks = detect_and_segment(det_rgb, device, args.fitting_prompt, args.gdino_threshold)
            fit_xyz = masks_to_xyz(rec, det_img, fit_masks)
            print(f"  fittings: {len(fit_masks)} detection(s) -> {len(fit_xyz)} 3D points")
            elec_boxes, elec_masks = detect_and_segment(det_rgb, device, args.electrical_prompt, args.gdino_threshold)
            elec_xyz = masks_to_xyz(rec, det_img, elec_masks)
            print(f"  electricals: {len(elec_masks)} detection(s) -> {len(elec_xyz)} 3D points")
            window_boxes, window_masks = detect_and_segment(det_rgb, device, args.window_prompt, args.gdino_threshold)
            window_box_viz, window_mask_viz = None, None
            if window_masks:
                img_area = det_rgb.shape[0] * det_rgb.shape[1]
                plausible_idx = [i for i, m in enumerate(window_masks) if m.astype(bool).sum() < 0.5 * img_area]
                plausible = [window_masks[i] for i in plausible_idx]
                window_xyz = masks_to_xyz(rec, det_img, plausible)
                print(f"  window: {len(plausible)}/{len(window_masks)} plausible detection(s) "
                      f"-> {len(window_xyz)} 3D points")
                if plausible_idx:
                    areas = [(window_boxes[i][2] - window_boxes[i][0]) * (window_boxes[i][3] - window_boxes[i][1])
                            for i in plausible_idx]
                    best_i = plausible_idx[int(np.argmax(areas))]
                    window_box_viz, window_mask_viz = window_boxes[best_i], window_masks[best_i]

            save_segmentation_viz(det_bgr, args.sfm_dir / "gdino_segmentation.png",
                                  elec_boxes=elec_boxes, elec_masks=elec_masks,
                                  window_box=window_box_viz, window_mask=window_mask_viz,
                                  pipe_boxes=pipe_boxes, pipe_masks=pipe_masks,
                                  fitting_boxes=fit_boxes, fitting_masks=fit_masks,
                                  ruck_px=ruck_px)
            print(f"Saved {args.sfm_dir / 'gdino_segmentation.png'}")

            if args.depth_model != "none":
                sys.path.insert(0, str(_ROOT / "08_depth"))
                from depth_models import (load_metric_anything, predict_metric_anything,
                                          load_depthpro, predict_hf)
                loaders = {"metric_anything": (load_metric_anything, predict_metric_anything),
                          "depthpro": (load_depthpro, predict_hf)}
                load_fn, predict_fn = loaders[args.depth_model]
                dm = load_fn(device)
                h, w = det_rgb.shape[:2]
                depth_cm = predict_fn(det_rgb, dm, device, focal_px=float(w))
                save_depth_viz(depth_cm, args.sfm_dir / "depth_map.png")
                print(f"Saved {args.sfm_dir / 'depth_map.png'} ({args.depth_model})")

                if not args.no_densify_depth:
                    cam_det = rec.cameras[det_img.camera_id]
                    P_det = np.asarray(det_img.cam_from_world().matrix())
                    R_det, t_det = P_det[:, :3], P_det[:, 3]
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
                    if len(sfm_z) >= 15:
                        ratios = np.array(sfm_z) / np.array(mono_z)
                        med = float(np.median(ratios))
                        spread = float(np.median(np.abs(ratios - med)) / max(med, 1e-6))
                        if spread < 0.25:
                            depth_aligned = depth_cm * med
                            h_img, w_img = depth_aligned.shape
                            stride = args.densify_stride
                            ys, xs = np.mgrid[0:h_img:stride, 0:w_img:stride]
                            ys, xs = ys.ravel(), xs.ravel()
                            dvals = depth_aligned[ys, xs]
                            valid = np.isfinite(dvals) & (dvals > 20) & (dvals < 800)
                            ys, xs, dvals = ys[valid], xs[valid], dvals[valid]
                            pix = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
                            rays = cam_det.cam_from_img(pix)
                            Xc_pts_cm = np.concatenate([rays * dvals[:, None], dvals[:, None]], axis=1)
                            Xc_units = Xc_pts_cm / cm_per_unit
                            dense_pts_model = (Xc_units - t_det) @ R_det
                            print(f"Depth densification: {len(sfm_z)} co-visible calibration points, "
                                  f"scale ratio {med:.2f} (spread {spread * 100:.0f}%) — accepted, "
                                  f"{len(dense_pts_model)} dense points added")
                        else:
                            print(f"Depth densification: scale ratio too inconsistent across "
                                  f"{len(sfm_z)} co-visible points (spread {spread * 100:.0f}%) — "
                                  f"skipping, sparse SfM points only")
                    else:
                        print(f"Depth densification: only {len(sfm_z)} co-visible calibration "
                              f"points — too few to trust the scale alignment, skipping")

            if not args.no_wall_seg:
                # Segmentation itself always runs (diagnostic viz, same
                # standing as depth_map.png/gdino_segmentation.png) —
                # whether it GATES the RANSAC pool is a separate decision
                # (--wall-seg-hard-gate). Hard-gating regressed the
                # flagship Klaus_Rombergg case even alone, no densify
                # involved: restricting the pool to ONLY mask-lifted points
                # cost the previously-winning wall enough native points
                # (imperfect segmentation boundary, real wall points near
                # edges/equipment the model didn't confidently label) that
                # it dropped from 3252 to under 500 inliers and failed the
                # trust floor outright. Off by default until that's fixed.
                wall_mask = segment_wall_mask(det_rgb, device, args.wall_seg_model)
                wall_seg_xyz_raw = masks_to_xyz(rec, det_img, [wall_mask.astype(np.uint8)])
                print(f"Wall segmentation (ADE20K): {wall_mask.mean() * 100:.0f}% of frame -> "
                      f"{len(wall_seg_xyz_raw)} sparse SfM points lifted")
                save_wall_seg_viz(det_bgr, wall_mask, args.sfm_dir / "wall_segmentation.png")
                print(f"Saved {args.sfm_dir / 'wall_segmentation.png'}")
                if args.wall_seg_hard_gate:
                    wall_seg_xyz = wall_seg_xyz_raw
        else:
            print(f"GDINO+SAM2 obstacle detection skipped: could not load frame {det_frame_name}")

    pts = np.asarray([p.xyz for p in rec.points3D.values()
                      if p.track.length() >= 3 and p.error < 1.5])
    cams = np.array([im.projection_center() for im in rec.images.values()])
    W, H = args.unit_wh_cm

    def search_wall(origin, u_ax, v_ax, n_ax, n_inl, wall_idx, allow_window_hole: bool):
        """Full obstacle/window/candidate-grid search on ONE wall plane.
        Returns a list of valid candidate dicts (each tagged with which
        wall it came from), or [] if unusable."""
        d_plane = (pts - origin) @ n_ax * cm_per_unit
        uv = np.stack([(pts - origin) @ u_ax, (pts - origin) @ v_ax], axis=1) * cm_per_unit
        front_sign = np.sign(float(np.median((cams - origin) @ n_ax)))
        d_front = d_plane * front_sign

        wall_band = np.abs(d_plane) < 8.0
        obst_band = (d_front >= 8.0) & (d_front < 40.0)
        uv_wall = uv[wall_band]
        uv_obst = density_filter(uv[obst_band])

        # Densified depth points feed WALL-SURFACE evidence (extent + the
        # trust-floor's "is this really a flat, well-covered wall" signal)
        # only — never obstacle/hole classification. Per-pixel monocular
        # depth noise is coarse enough that points genuinely on the wall
        # can scatter a few cm into the 8-40cm "in front" obstacle band,
        # and at dense-point volume that's enough noise to saturate the
        # whole grid with false obstacles (seen on Wolfram_Koestler: three
        # walls newly cleared the RANSAC evidence floor after densification,
        # but every one of them still produced 0 valid candidates — the
        # noise was misread as obstacles covering the entire wall). GDINO
        # masks and the native sparse SfM cloud already cover obstacles
        # reliably; densification only needs to answer "is there enough
        # wall surface here to trust the 5cm grid," which tolerates a few
        # cm of depth noise just fine.
        if len(dense_pts_model):
            dz = (dense_pts_model - origin) @ n_ax * cm_per_unit
            duv = np.stack([(dense_pts_model - origin) @ u_ax,
                            (dense_pts_model - origin) @ v_ax], axis=1) * cm_per_unit
            dense_wall = duv[np.abs(dz) < 8.0]
            if len(dense_wall):
                uv_wall = np.concatenate([uv_wall, dense_wall])
        # Window/recess keep-out from the SfM cloud alone is only meaningful
        # when the plane IS the actual wall surface (triangulated marker).
        # For a RANSAC plane sliced through the cloud, points behind it are
        # mostly deeper wall, not recesses — so skip it there (the GDINO
        # window detection below still applies either way).
        if allow_window_hole:
            hole_band = (d_front <= -8.0) & (d_front > -80.0)
            uv_hole = density_filter(uv[hole_band])
        else:
            uv_hole = np.empty((0, 2))

        elec_uv = xyz_to_uv(elec_xyz, origin, u_ax, v_ax, cm_per_unit)
        for xyz in (pipe_xyz, fit_xyz, elec_xyz):
            extra = xyz_to_uv(xyz, origin, u_ax, v_ax, cm_per_unit)
            if len(extra):
                uv_obst = np.concatenate([uv_obst, density_filter(extra)])
        uv_win = xyz_to_uv(window_xyz, origin, u_ax, v_ax, cm_per_unit)
        if len(uv_win):
            uv_hole = np.concatenate([uv_hole, density_filter(uv_win)]) if len(uv_hole) else density_filter(uv_win)

        if len(uv_wall) < 100:
            print(f"  wall[{wall_idx}]: WARNING sparse wall coverage ({len(uv_wall)} pts) — "
                  f"placement area may be underestimated")

        uv_evidence = np.concatenate([uv_wall, uv_obst] + ([uv_hole] if len(uv_hole) else []))
        if len(uv_evidence) < 20:
            print(f"  wall[{wall_idx}]: not enough wall evidence, skipping")
            return []
        lo = np.percentile(uv_evidence, 1, axis=0)
        hi = np.percentile(uv_evidence, 99, axis=0)

        origin_z = float(((origin * cm_per_unit) @ R_up.T)[2])
        v_floor = z_floor - origin_z
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
                    if int(in_hole.sum()) >= 3:
                        n_hole_rej += 1
                        continue
                if len(uv_obst):
                    dx = np.maximum(np.abs(uv_obst[:, 0] - cu) - W / 2, 0)
                    dy = np.maximum(np.abs(uv_obst[:, 1] - cv_) - H / 2, 0)
                    d_obst = np.hypot(dx, dy)
                    if int(np.sum(d_obst == 0.0)) >= 3:
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
        ruck_str = f"ruck u={ruck_uv[0]:.0f}cm v={ruck_uv[1]:.0f}cm" if ruck_uv is not None else "no Rücklauf ref"
        print(f"  wall[{wall_idx}] ({n_inl} inl): {n_cells} cells, {n_hole_rej} window-rejected, "
              f"{n_obst_rej} obstacle-rejected, {len(cands_out)} valid "
              f"(wall-pts={len(uv_wall)}, {ruck_str})")
        return cands_out

    all_cands = []
    if scale_info.get("segments"):
        origin, u_ax, v_ax, n_ax = marker_wall_frame(scale_info, up)
        wall_tilt = float(np.degrees(np.arcsin(abs(n_ax @ up))))
        print(f"Wall plane [triangulated marker]: origin={np.round(origin, 3)}, "
              f"tilt from vertical: {wall_tilt:.1f} deg")
        if wall_tilt > 25.0:
            print(f"FAILED: wall plane is {wall_tilt:.0f}° from vertical — unreliable; skipping.")
            return
        if ruck_X is not None:
            ruck_dist_wall = float((ruck_X - origin) @ n_ax) * cm_per_unit
            print(f"Rücklauf 3D (model units): {np.round(ruck_X, 3)} ({abs(ruck_dist_wall):.0f} cm from wall plane)")
        # The triangulated marker IS the real wall surface, precisely —
        # trust it unconditionally, no RANSAC evidence-quality question here.
        all_cands = search_wall(origin, u_ax, v_ax, n_ax, n_inl=10**9, wall_idx=0, allow_window_hole=True)
    else:
        pts_model_full = load_filtered_points(rec)
        if len(wall_seg_xyz) >= 50:
            pts_model = wall_seg_xyz
            print(f"Wall-seg-gated RANSAC pool: {len(pts_model)} points "
                  f"(of {len(pts_model_full)} total sparse)")
        else:
            pts_model = pts_model_full
            if args.wall_seg_hard_gate:
                print("Wall segmentation gate: too few lifted points — searching full sparse cloud")
        if len(dense_pts_model):
            pts_model = np.concatenate([pts_model, dense_pts_model])
        # Without a Rücklauf anchor, search the whole room instead of a
        # 250cm neighborhood around one point — wall_frame_ransac_multi
        # falls back to the full floor-band pool whenever the near_cm
        # filter leaves too few points, so a deliberately huge near_cm
        # here just disables the proximity restriction outright.
        wall_ref = ruck_X if ruck_X is not None else pts_model.mean(axis=0)
        wall_near_cm = 250.0 if ruck_X is not None else 1e6
        # Sparse SfM points rarely form a tight false-positive line fit
        # from a short object the way a dense per-pixel cloud can — this
        # stays a low sanity floor here (not the 100cm default) so as not
        # to disturb already-validated sparse-cloud wall selection.
        candidates_wf = wall_frame_ransac_multi(pts_model, up, wall_ref, z_floor, cm_per_unit,
                                                near_cm=wall_near_cm, min_height_extent_cm=20.0)
        if not candidates_wf:
            print("FAILED: no dominant vertical wall found")
            return
        print(f"Wall candidates found: {len(candidates_wf)}")
        by_inliers = sorted(candidates_wf, key=lambda c: -c["n_inl"])[:args.max_walls_searched]
        by_angle = sorted(candidates_wf, key=lambda c: c["view_angle"])[:args.max_walls_searched]
        seen_ids, search_set = set(), []
        for c in by_angle + by_inliers:
            if id(c) not in seen_ids:
                seen_ids.add(id(c))
                search_set.append(c)
        print(f"Searching {len(search_set)}/{len(candidates_wf)} wall candidates "
              f"(union of top-{args.max_walls_searched} by angle and by inliers):")
        for wall_idx, wall in enumerate(search_set):
            wall_tilt = float(np.degrees(np.arcsin(abs(wall["n"] @ up))))
            if wall_tilt > 25.0 or wall["view_angle"] > 60.0:
                continue
            all_cands.extend(search_wall(wall["origin"], wall["u"], wall["v"], wall["n"],
                                         wall["n_inl"], wall_idx, allow_window_hole=False))

    if not all_cands:
        print("FAILED: no valid placement found on any wall candidate")
        return

    trustworthy = [c for c in all_cands if c["wall_n_inl"] >= args.min_wall_inliers]
    if not trustworthy:
        print(f"FAILED: {len(all_cands)} candidate(s) found, but none on a wall with "
              f"--min-wall-inliers {args.min_wall_inliers} — too little evidence to trust "
              f"the obstacle math. Lower the threshold to see them anyway.")
        return
    cands = trustworthy

    cands.sort(key=lambda c: c["score"])
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

    # everything downstream (overlay projection, JSON) uses the WINNING
    # candidate's own wall frame — candidates can come from different walls
    origin = np.array(top[0]["wall_origin"])
    u_ax = np.array(top[0]["wall_u"])
    v_ax = np.array(top[0]["wall_v"])
    n_ax = np.array(top[0]["wall_n"])

    # ── overlay: prefer the frame GDINO already detected obstacles on ──────
    # Real failure: picking "whichever registered frame has the recommended
    # point most centered in-bounds" found a frame that sees the same wall
    # through a doorway/curtain from across an ADJACENT ROOM — technically
    # in-bounds and centered, but occluded and misleading. No occlusion
    # check existed (same class of bug 07_pipes/pipe_paths.py already had
    # to fix for its own cluster visualization). The detection frame is
    # known-good — it's the one an object detector already looked at and
    # found real pipes/fittings/electricals on — so use it directly instead
    # of searching for a "best" frame that can silently pick an occluded one.
    best_center = np.array(top[0]["center_model"])

    # Real failure this was built to catch: neither "sees the point in
    # bounds" nor "most centered" checks whether the camera actually faces
    # the WALL — a frame that grazes along a wall can see the recommended
    # point dead-center and still render the rectangle as a self-
    # intersecting "bowtie" of crossed lines (seen on Albert_Mayer_IMG_1831,
    # Christian_Heimes_IMG_4716, Monika_Mulock_IMG_6278 in a 24-video batch,
    # all via the old "NOT occlusion-checked" fallback). A first attempt at
    # gating this by face-on ANGLE alone wasn't enough — Albert_Mayer and
    # Christian_Heimes still rendered garbage from their best-angle frame,
    # because extreme grazing incidence at close range distorts the
    # projection into a bowtie well before any fixed angle threshold is
    # crossed. Testing the actual projected quad's winding order catches
    # that directly instead of guessing at an angle cutoff.
    corners_3d = [origin + ((top[0]["u_cm"] + du) * u_ax + (top[0]["v_cm"] + dv) * v_ax) / cm_per_unit
                  for du, dv in [(-W / 2, -H / 2), (W / 2, -H / 2), (W / 2, H / 2), (-W / 2, H / 2)]]

    def project_in(im, X_model):
        cam_i = rec.cameras[im.camera_id]
        P_i = np.asarray(im.cam_from_world().matrix())
        Xc = P_i @ np.append(X_model, 1.0)
        if Xc[2] <= 0:
            return None
        return np.asarray(cam_i.img_from_cam(Xc[None, :3] / Xc[2])).ravel()

    def quad_ok(im):
        """True if the winning rectangle projects to a convex, non-sliver,
        FULLY-IN-FRAME quadrilateral in this camera — rejects both the
        self-intersecting 'bowtie' render a pure face-on-angle check can
        miss, and a convex-but-partly-off-frame render (a quad can pass
        the convexity/area test with 1-3 corners outside the visible
        image if only the rectangle's CENTER was checked in-bounds —
        seen on Monika_Mulock_IMG_6278, where the box still looked like a
        line running off the top of the frame after the convexity fix)."""
        cam_i = rec.cameras[im.camera_id]
        w, h = cam_i.width, cam_i.height
        pts = []
        for X in corners_3d:
            p = project_in(im, X)
            if p is None or not (0 <= p[0] < w and 0 <= p[1] < h):
                return False
            pts.append(p)
        pts = np.array(pts)
        signs = []
        for i in range(4):
            a, b, c = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
            cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            signs.append(np.sign(cross))
        if len(set(s for s in signs if s != 0)) > 1:
            return False
        area = 0.5 * abs(sum(pts[i][0] * pts[(i + 1) % 4][1] - pts[(i + 1) % 4][0] * pts[i][1]
                             for i in range(4)))
        xs, ys = pts[:, 0], pts[:, 1]
        bbox = (xs.max() - xs.min()) * (ys.max() - ys.min())
        return bbox >= 1 and (area / bbox) >= 0.15

    def sees_point_inbounds(im, X_model):
        cam_i = rec.cameras[im.camera_id]
        P_i = np.asarray(im.cam_from_world().matrix())
        Xc = P_i @ np.append(X_model, 1.0)
        if Xc[2] <= 0:
            return False
        xy = np.asarray(cam_i.img_from_cam(Xc[None, :3] / Xc[2])).ravel()
        w, h = cam_i.width, cam_i.height
        return 0 <= xy[0] < w and 0 <= xy[1] < h

    view_img = None
    preferred_name = det_frame_name or ruck_frame
    if preferred_name:
        cand = next((im for im in rec.images.values() if im.name == preferred_name), None)
        if cand is not None and sees_point_inbounds(cand, best_center) and quad_ok(cand):
            view_img = cand
            print(f"Overlay frame: {preferred_name} (the detection frame — known to show the real equipment)")
    if view_img is None:
        # Fallback: search all frames for in-bounds + a clean projected
        # quad, pick most centered among those. Still not occlusion-checked.
        view_score = -1.0
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
            if not quad_ok(im):
                continue
            centered = 1.0 - (abs(xy[0] - w / 2) / w + abs(xy[1] - h / 2) / h)
            if centered > view_score:
                view_score, view_img = centered, im
        if view_img is not None:
            print(f"Overlay frame: {view_img.name} (fallback search, quad-shape-filtered — "
                  f"NOT occlusion-checked, verify visually)")
    if view_img is None:
        # Last resort: no in-bounds frame renders a clean quad at all —
        # fall back to unfiltered "most centered" so SOMETHING renders,
        # but say so plainly since it may still be a degenerate sliver.
        view_score = -1.0
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
            centered = 1.0 - (abs(xy[0] - w / 2) / w + abs(xy[1] - h / 2) / h)
            if centered > view_score:
                view_score, view_img = centered, im
        if view_img is not None:
            print(f"Overlay frame: {view_img.name} (WARNING: no frame renders a clean quad for this "
                  f"wall — every registered view is too grazing; render may be a degenerate sliver, "
                  f"treat the JSON coordinates as authoritative, not this image)")
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
        # Each candidate's (u_cm, v_cm) is expressed in ITS OWN wall's frame
        # (c["wall_origin"]/wall_u/wall_v — candidates can legitimately come
        # from different walls, the dedup logic above explicitly allows it).
        # Using the single global origin/u_ax/v_ax (top[0]'s wall) for every
        # candidate here was a real bug: any #2/#3 from a different wall got
        # its local coordinates reinterpreted through the WRONG wall's
        # frame, producing a corrupted 3D position — not necessarily even
        # camera-adjacent, just wrong. Only #1 was ever guaranteed correct
        # (its own wall IS the global frame by construction).
        c_origin = np.array(c["wall_origin"])
        c_u = np.array(c["wall_u"])
        c_v = np.array(c["wall_v"])
        corners = []
        for du, dv in [(-W/2, -H/2), (W/2, -H/2), (W/2, H/2), (-W/2, H/2)]:
            X = c_origin + ((c["u_cm"] + du) * c_u + (c["v_cm"] + dv) * c_v) / cm_per_unit
            p = project(X)
            if p is None:
                corners = []
                break
            corners.append(p)
        if corners:
            cv2.polylines(bgr, [np.array(corners)], True, colors[i % 3], 3)
            label = (f"#{i+1} d={c['dist_to_rucklauf_cm']:.0f}cm" if c.get("dist_to_rucklauf_cm") is not None
                     else f"#{i+1} clr={c['clearance_cm']:.0f}cm")
            cv2.putText(bgr, label, (corners[0][0], corners[0][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i % 3], 2, cv2.LINE_AA)
    if ruck_X is not None:
        rp = project(ruck_X)
        if rp:
            cv2.drawMarker(bgr, rp, (255, 0, 0), cv2.MARKER_STAR, 26, 3)
            cv2.putText(bgr, "Ruecklauf", (rp[0] + 12, rp[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    out_img = args.sfm_dir / "placement_3d.jpg"
    cv2.imwrite(str(out_img), bgr)

    out = dict(cm_per_unit=cm_per_unit,
               unit_wh_cm=[W, H],
               rucklauf=dict(model_xyz=ruck_X.tolist() if ruck_X is not None else None, frame=ruck_frame,
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
