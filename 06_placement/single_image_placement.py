"""
single_image_placement.py — placement recommendation from ONE photo, no
video/SfM required.

Same task as placement_3d.py (find free wall space for the indoor unit, near
the Rücklauf and electricals, away from windows), but the wall geometry comes
from a single monocular depth prediction instead of a triangulated SfM cloud.
Reuses placement_3d.py's wall-plane RANSAC and room_dims.py's gravity-frame
helpers directly — the candidate-grid scoring only cares about "points near a
vertical wall plane in cm", not where they came from.

Known limitation, stated up front rather than hidden: single-view depth has
no camera-pose diversity, so "up" can't be measured from multiple camera
positions the way placement_3d.py does. This assumes a LEVEL CAMERA (image
y-axis = gravity), the same simplifying assumption metric_anything_room_dims.py
already documents and uses for its own single-frame estimates. A tilted phone
will skew results — this is an MVP, not a replacement for the SfM-based path
when a full video is available.

Usage:
  python 06_placement/single_image_placement.py IMG_1234.jpg \
      --depth-model metric_anything --unit-wh-cm 60 40

  # manual overrides instead of auto-detection:
  python 06_placement/single_image_placement.py IMG_1234.jpg \
      --depth-model depthpro \
      --rucklauf-px 640 360 --electrical-px 200 300 --window-box 900 100 1200 400
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "05_geometry"))
sys.path.insert(0, str(_ROOT / "08_depth"))
sys.path.insert(0, str(_ROOT / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from room_dims import rotation_to_z, density_peak
from depth_models import (
    load_metric_anything, predict_metric_anything,
    load_depthpro, predict_hf,
)
from placement_3d import blue_circle_candidates, segment_wall_mask, save_wall_seg_viz

DEPTH_LOADERS = {
    "metric_anything": (load_metric_anything, predict_metric_anything),
    "depthpro": (load_depthpro, predict_hf),
}


def wall_frame_ransac_multi(pts_model, up_world, ref_model, z_floor_cm, cm_per_unit,
                            iters: int = 6000, thresh_cm: float = 6.0, near_cm: float = 250.0,
                            min_inlier_frac: float = 0.05):
    """Same 2D line-RANSAC-in-gravity-aligned-top-down-projection
    placement_3d.py's wall_frame_ransac() uses, but tracks every distinct
    well-supported candidate line instead of keeping only the single
    best-inlier one, then picks whichever is most face-on to the camera.

    Why: placement_3d.py's SfM-derived RANSAC sees a wall built from many
    camera positions across a whole video, so "most inliers" is already a
    good proxy for "the real wall". A single photo has no such redundancy —
    a narrow/angled shot often has 2-3 real vertical surfaces in frame (the
    equipment wall, a side wall, a recess), and majority-vote reliably picks
    whichever has more raw point density, not whichever the camera is
    actually facing. Evaluating multiple candidates and preferring the
    face-on one is a much better prior for this case, and lets `near_cm` be
    generous (reaching real free space farther from the Rücklauf) without
    the wrong-wall risk that forced a tight radius before.

    Returns a list of dicts (each: origin, u, v, n, n_inl, view_angle),
    sorted most-face-on first, or None if nothing reaches the minimum
    inlier count at all.
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

    # Dedup by LINE GEOMETRY (angle mod 180 deg + perpendicular offset from
    # the origin), not by inlier-set overlap. Inlier-set overlap fails on a
    # single long wall: different random samples along its length each
    # explain a different sub-region with LOW pairwise overlap despite
    # being the same physical surface, which used to survive dedup as
    # dozens of fake "distinct" candidates that were really one wall
    # sampled repeatedly, starving out genuinely different (smaller, e.g.
    # side-wall) surfaces from ever being tracked.
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
        candidates.append(dict(origin=origin, u=u, v=v, n=n, n_inl=c, view_angle=view_angle))

    candidates.sort(key=lambda cd: cd["view_angle"])
    return candidates if candidates else None


def backproject(depth_cm: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                stride: int = 4):
    """Full-frame depth map -> camera-frame 3D points (cm), plus the source
    pixel (row, col) each point came from — kept so window/electrical
    detections in image space can be matched to specific 3D points later.
    Convention matches metric_anything_room_dims.py: X=right, Y=down (image
    convention), Z=forward/depth; 'up' is therefore (0,-1,0)."""
    h, w = depth_cm.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    d = depth_cm[0:h:stride, 0:w:stride]
    valid = np.isfinite(d) & (d > 0)
    X = (xs - cx) / fx * d
    Y = (ys - cy) / fy * d
    pts = np.stack([X[valid], Y[valid], d[valid]], axis=1)
    pix = np.stack([ys[valid], xs[valid]], axis=1)   # (row, col)
    return pts, pix


def project_to_pixel(pt_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float):
    X, Y, Z = pt_cam
    if Z <= 1e-6:
        return None
    return (cx + X / Z * fx, cy + Y / Z * fy)


_GDINO_SAM2_CACHE = {}


def load_gdino_sam2(device: str, gdino_checkpoint: str = "IDEA-Research/grounding-dino-tiny"):
    """Loaded once per checkpoint, reused across calls. Default matches the
    rest of the repo (experiment_pipeline.py, compare_arms.py,
    pipe_paths.py) — pass a bigger checkpoint (e.g. grounding-dino-base) via
    --gdino-checkpoint for better recall than sweep_prompts.py's tiny-model
    ceiling (R=0.525 on 'pipe'), without touching that shared baseline."""
    if gdino_checkpoint not in _GDINO_SAM2_CACHE:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from experiment_pipeline import load_sam2
        proc = AutoProcessor.from_pretrained(gdino_checkpoint)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_checkpoint).to(device)
        model.eval()
        print(f"  GroundingDINO loaded ({gdino_checkpoint})")
        sam2 = _GDINO_SAM2_CACHE.get("_sam2") or load_sam2(device)
        _GDINO_SAM2_CACHE["_sam2"] = sam2
        _GDINO_SAM2_CACHE[gdino_checkpoint] = (proc, model, sam2)
    return _GDINO_SAM2_CACHE[gdino_checkpoint]


def detect_and_segment(image_rgb, device, prompt: str, threshold: float,
                       gdino_checkpoint: str = "IDEA-Research/grounding-dino-tiny"):
    """GDINO boxes -> SAM2 masks, same box->mask pairing the rest of the repo
    uses. `prompt` may be a single phrase or a list of phrases to ensemble
    (union of detections, NMS-deduplicated) — useful when one phrasing alone
    misses real objects a synonym catches (e.g. 'pipe' vs 'conduit' vs
    'metal tube'), which sweep_prompts.py's per-phrase evaluation never
    tried combining. Returns (boxes_xyxy, masks), masks a list of HxW bool
    arrays in the same order as boxes."""
    from experiment_pipeline import detect_gdino, sam2_from_boxes
    gdino_proc, gdino_model, sam2 = load_gdino_sam2(device, gdino_checkpoint)
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    all_boxes, all_scores = [], []
    for p in prompts:
        boxes, labels, scores = detect_gdino(image_rgb, gdino_proc, gdino_model, p, threshold, device)
        all_boxes.append(boxes)
        all_scores.extend(scores)
    boxes = np.concatenate(all_boxes, axis=0) if any(len(b) for b in all_boxes) else np.empty((0, 4))
    if len(prompts) > 1 and len(boxes) > 1:
        boxes = nms_dedup(boxes, np.array(all_scores))
    masks = sam2_from_boxes(sam2, image_rgb, boxes) if len(boxes) else []
    return boxes, masks


def nms_dedup(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.6) -> np.ndarray:
    """Simple greedy NMS to deduplicate the same real object caught by
    multiple prompts in an ensemble (e.g. 'pipe' and 'metal tube' both
    firing on the same physical pipe)."""
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(area_i + area_r - inter, 1e-9)
        order = rest[iou < iou_thresh]
    return boxes[keep]


def detect_electricals(image_rgb, device, prompt, threshold: float, gdino_checkpoint: str):
    boxes, masks = detect_and_segment(image_rgb, device, prompt, threshold, gdino_checkpoint)
    return boxes, masks


def detect_window(image_rgb, device, prompt, threshold: float, gdino_checkpoint: str):
    boxes, masks = detect_and_segment(image_rgb, device, prompt, threshold, gdino_checkpoint)
    if len(boxes) == 0:
        return None, None
    img_area = image_rgb.shape[0] * image_rgb.shape[1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # a "window" spanning most of the frame is a GDINO false positive on the
    # whole scene, not a real window — reject rather than treat the entire
    # photo as keep-out
    plausible = areas < 0.5 * img_area
    if not plausible.any():
        return None, None
    idx = np.where(plausible)[0]
    best = idx[int(np.argmax(areas[idx]))]
    return boxes[best], masks[best]   # largest remaining window box + its mask


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
    """GDINO detections + SAM2 masks, colored per class, matching how the
    rest of the repo visualizes box->mask segmentation (pipe_paths.py etc.)."""
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


def main():
    parser = argparse.ArgumentParser(description="Single-image placement recommendation")
    parser.add_argument("image", type=Path)
    parser.add_argument("--depth-model", choices=["metric_anything", "depthpro"], required=True)
    parser.add_argument("--focal-px", type=float, default=None,
                        help="Default: image width (MetricAnything's own no-calibration fallback)")
    parser.add_argument("--unit-wh-cm", type=float, nargs=2, default=[60.0, 40.0])
    parser.add_argument("--min-clearance-cm", type=float, default=20.0,
                        help="Raised from an initial 10cm after a real failure: a candidate scored "
                             "11.8cm clearance and still landed directly on an undetected valve/trap "
                             "cluster — 10cm wasn't a meaningful margin against an incomplete obstacle "
                             "set. 20cm is a more honest floor given detection isn't exhaustive.")
    parser.add_argument("--preferred-height-cm", type=float, default=120.0)
    parser.add_argument("--min-bottom-cm", type=float, default=20.0)
    parser.add_argument("--w-dist", type=float, default=1.0)
    parser.add_argument("--w-elec", type=float, default=1.0)
    parser.add_argument("--w-clear", type=float, default=3.0)
    parser.add_argument("--w-height", type=float, default=0.5)
    parser.add_argument("--rucklauf-px", type=float, nargs=2, default=None,
                        help="Manual override: pixel (x y) instead of auto blue-cap detection")
    parser.add_argument("--rucklauf-fallback-prompt", default="pressure gauge on pipe . pipe valve",
                        help="GDINO prompt tried when no blue cap is found — some installations mark "
                             "Vorlauf/Rücklauf with a gauge pair instead of a colored cap. Only reports "
                             "candidates; never auto-picks which one is Rücklauf, since nothing in the "
                             "image alone says which.")
    parser.add_argument("--electrical-px", type=float, nargs=2, action="append", default=None,
                        help="Manual override: pixel (x y), repeatable, instead of GDINO auto-detect")
    parser.add_argument("--electrical-prompt", default="electrical panel . fuse box . circuit breaker")
    parser.add_argument("--window-box", type=float, nargs=4, default=None,
                        help="Manual override: x0 y0 x1 y1 instead of GDINO auto-detect")
    parser.add_argument("--window-prompt", default="window")
    parser.add_argument("--pipe-prompt", nargs="+", default=["pipe"],
                        help="GDINO prompt(s) for pipe segmentation. 'pipe' at threshold 0.25 is the "
                             "single-phrase best in this repo (experiments/sweep_prompts.py, R=0.525), "
                             "but pass multiple phrases (e.g. pipe 'metal pipe' 'heating pipe' conduit "
                             "tube) to ensemble — union of detections, NMS-deduped — for better recall "
                             "than any one phrasing alone. Pipe masks feed obstacle-avoidance scoring, "
                             "not just the visualization — pass --no-pipes to skip.")
    parser.add_argument("--pipe-threshold", type=float, default=0.25)
    parser.add_argument("--no-pipes", action="store_true", help="Skip pipe detection entirely")
    parser.add_argument("--fitting-prompt", nargs="+",
                        default=["valve", "trap", "fitting", "gauge", "meter"],
                        help="GDINO prompt(s) for small fixtures (valves, traps, gauges) that are "
                             "thin/close enough to the wall to slip past BOTH the pipe detector and "
                             "the depth-based obstacle band — see the 'Small fittings' comment in "
                             "main() for the real failure this was added to catch. Folds into the "
                             "same obstacle-avoidance scoring as pipes, own color in the visualization.")
    parser.add_argument("--fitting-threshold", type=float, default=0.25)
    parser.add_argument("--no-fittings", action="store_true", help="Skip fitting detection entirely")
    parser.add_argument("--gdino-threshold", type=float, default=0.25)
    parser.add_argument("--gdino-checkpoint", default="IDEA-Research/grounding-dino-tiny",
                        help="Swap in 'IDEA-Research/grounding-dino-base' for better recall/precision "
                             "than the tiny checkpoint the rest of this repo uses (slower, more VRAM)")
    parser.add_argument("--stride", type=int, default=4, help="Depth-map backprojection pixel stride")
    parser.add_argument("--min-wall-inliers", type=int, default=2000,
                        help="Minimum RANSAC inlier count for a wall's candidates to be trusted "
                             "in ranking. A good angle doesn't mean good evidence — a thin point "
                             "cloud makes 5cm-grid obstacle math unreliable even when well-oriented "
                             "(real failure: a 1003-inlier wall put a candidate on a visible "
                             "electrical cluster its own obstacle points didn't project onto "
                             "precisely enough to catch).")
    parser.add_argument("--max-walls-searched", type=int, default=15,
                        help="Cap on how many distinct wall candidates get a full obstacle/grid "
                             "search — bounds runtime. Capped by inlier count so well-evidenced "
                             "walls (even less face-on ones, e.g. a real side wall) aren't dropped "
                             "in favor of near-duplicate re-fits of the single dominant wall.")
    parser.add_argument("--wall-near-cm", type=float, default=250.0,
                        help="RANSAC pool radius around the Rücklauf — tighter helps disambiguate "
                             "between multiple visible wall orientations in one shot")
    parser.add_argument("--wall-band-cm", type=float, default=8.0,
                        help="How close (perpendicular cm) a point must be to the fitted plane to "
                             "count as 'wall surface'. Copied from placement_3d.py's SfM context, "
                             "where the point cloud is geometrically precise — monocular depth's "
                             "absolute-scale drift across a wide image may need this looser (try "
                             "15-20) to avoid under-estimating how much open wall is actually there.")
    parser.add_argument("--no-wall-seg", action="store_true",
                        help="Skip ADE20K wall segmentation diagnostic (wall_segmentation.png).")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bgr = cv2.imread(str(args.image))
    if bgr is None:
        print(f"FAILED: could not read {args.image}")
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = bgr.shape[:2]
    fx = fy = args.focal_px or float(w)
    cx, cy = w / 2.0, h / 2.0
    print(f"Image {w}x{h}, f_px={fx:.0f} ({'given' if args.focal_px else 'width fallback, no calibration'})")

    out_dir = args.out or (args.image.parent / f"{args.image.stem}_placement")
    out_dir.mkdir(parents=True, exist_ok=True)

    load_fn, predict_fn = DEPTH_LOADERS[args.depth_model]
    print(f"Loading {args.depth_model}...")
    model = load_fn(device)
    depth_cm = predict_fn(rgb, model, device, focal_px=fx)
    print(f"Depth range: {np.nanmin(depth_cm):.0f}-{np.nanmax(depth_cm):.0f} cm")
    save_depth_viz(depth_cm, out_dir / "depth_map.png")
    print(f"Saved {out_dir / 'depth_map.png'}")

    pts_cm, pix = backproject(depth_cm, fx, fy, cx, cy, stride=args.stride)
    print(f"Backprojected {len(pts_cm)} points (stride={args.stride})")

    up_world = np.array([0.0, -1.0, 0.0])   # level-camera assumption
    R_up = rotation_to_z(up_world)
    z_all = (pts_cm @ R_up.T)[:, 2]
    z_floor = density_peak(z_all, 0.5, 35.0)
    if z_floor is None:
        print("FAILED: could not localize floor level")
        return
    print(f"Floor at z={z_floor:.1f} cm (gravity-aligned, level-camera assumption)")

    # ── Rücklauf ──────────────────────────────────────────────────────────
    if args.rucklauf_px:
        ruck_px, ruck_py = args.rucklauf_px
        ruck_source = "manual"
    else:
        cands = blue_circle_candidates(bgr)
        if cands:
            best = max(cands, key=lambda c: c["circularity"] * c["area"])
            ruck_px, ruck_py = best["px"], best["py"]
            ruck_source = f"auto blue-cap ({len(cands)} candidates)"
        else:
            # No colored cap in frame — some installations mark Vorlauf/
            # Rücklauf with a gauge pair on the pipe fittings instead (real
            # case: Frank_Baumer_Heizungsanlage frame 0002). Fall back to
            # GDINO for that visual pattern. This finds CANDIDATES, not an
            # answer — gauges come in Vorlauf/Rücklauf pairs and nothing in
            # the image alone says which one is which, so this stops and
            # asks rather than guessing (see the actual failure this was
            # built to avoid: silently picking a pixel and treating it as
            # real data).
            gauge_boxes, _ = detect_and_segment(rgb, device, args.rucklauf_fallback_prompt,
                                                args.gdino_threshold, args.gdino_checkpoint)
            if len(gauge_boxes) == 0:
                print("FAILED: no Rücklauf blue-cap found, and no gauge/valve fallback candidates "
                      "either — pass --rucklauf-px")
                return
            print(f"No blue cap found. Gauge/valve fallback (GDINO "
                  f"'{args.rucklauf_fallback_prompt}') found {len(gauge_boxes)} candidate(s):")
            for i, b in enumerate(gauge_boxes):
                cx_b, cy_b = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                print(f"  [{i}] px=({cx_b:.0f},{cy_b:.0f})  box={np.round(b).tolist()}")
            print("Can't tell which one is Rücklauf vs Vorlauf from the image alone — "
                  "rerun with --rucklauf-px <x> <y> using one of the candidates above.")
            return
    ruck_d = float(depth_cm[int(round(ruck_py)), int(round(ruck_px))])
    ruck_X = np.array([(ruck_px - cx) / fx * ruck_d, (ruck_py - cy) / fy * ruck_d, ruck_d])
    print(f"Rücklauf [{ruck_source}]: px=({ruck_px:.0f},{ruck_py:.0f}) depth={ruck_d:.0f}cm")

    # ── Electricals ───────────────────────────────────────────────────────
    elec_pts = []
    elec_boxes, elec_masks = None, None
    if args.electrical_px:
        for ex, ey in args.electrical_px:
            d = float(depth_cm[int(round(ey)), int(round(ex))])
            elec_pts.append(np.array([(ex - cx) / fx * d, (ey - cy) / fy * d, d]))
        print(f"Electricals [manual]: {len(elec_pts)} point(s)")
    else:
        elec_boxes, elec_masks = detect_electricals(rgb, device, args.electrical_prompt, args.gdino_threshold, args.gdino_checkpoint)
        for b in elec_boxes:
            ex, ey = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            d = float(depth_cm[int(round(ey)), int(round(ex))])
            elec_pts.append(np.array([(ex - cx) / fx * d, (ey - cy) / fy * d, d]))
        print(f"Electricals [auto GDINO '{args.electrical_prompt}']: {len(elec_pts)} detection(s)")

    # ── Window keep-out ───────────────────────────────────────────────────
    window_mask = None
    if args.window_box:
        window_box = np.array(args.window_box)
        window_source = "manual"
    else:
        window_box, window_mask = detect_window(rgb, device, args.window_prompt, args.gdino_threshold, args.gdino_checkpoint)
        window_source = f"auto GDINO '{args.window_prompt}'"
    if window_box is not None:
        print(f"Window [{window_source}]: box={np.round(window_box).tolist()}")
    else:
        print(f"Window [{window_source}]: none detected")

    # ── Pipes ─────────────────────────────────────────────────────────────
    # 'pipe' at threshold 0.25 is this repo's single best-validated GDINO
    # class (experiments/sweep_prompts.py: R=0.525, the primary detector
    # 07_pipes/pipe_paths.py already builds on) — unlike electrical/window
    # above, this isn't a first attempt at a new prompt. Pipe masks feed
    # obstacle-avoidance directly (see uv_obst below), not just the picture.
    pipe_boxes, pipe_masks = (None, None)
    if not args.no_pipes:
        pipe_boxes, pipe_masks = detect_and_segment(rgb, device, args.pipe_prompt, args.pipe_threshold, args.gdino_checkpoint)
        print(f"Pipes [auto GDINO '{args.pipe_prompt}']: {len(pipe_boxes)} detection(s)")

    # ── Small fittings (valves, traps, gauges) ───────────────────────────────
    # Added after a real failure: a top-ranked candidate on Dietmar_Baudisch
    # frame 0105 sat directly on a P-trap + valve + gauge cluster that NEITHER
    # the pipe prompt NOR the depth-based obstacle band caught — small,
    # geometrically thin fixtures close to the wall plane are exactly the
    # blind spot both detectors share. This is a second, independent GDINO
    # pass with its own vocabulary, folded into the same obstacle mechanism
    # as pipes below — not a fix for that specific miss, a real second layer.
    fitting_boxes, fitting_masks = (None, None)
    if not args.no_fittings:
        fitting_boxes, fitting_masks = detect_and_segment(rgb, device, args.fitting_prompt, args.fitting_threshold, args.gdino_checkpoint)
        print(f"Fittings [auto GDINO '{args.fitting_prompt}']: {len(fitting_boxes)} detection(s)")

    save_segmentation_viz(bgr, out_dir / "gdino_segmentation.png",
                          elec_boxes=elec_boxes, elec_masks=elec_masks,
                          window_box=window_box, window_mask=window_mask,
                          pipe_boxes=pipe_boxes, pipe_masks=pipe_masks,
                          fitting_boxes=fitting_boxes, fitting_masks=fitting_masks,
                          ruck_px=(ruck_px, ruck_py))
    print(f"Saved {out_dir / 'gdino_segmentation.png'}")

    # ── ADE20K wall segmentation (diagnostic only, matches placement_3d.py/
    # placement_depth.py's convention — NOT used to gate the RANSAC pool
    # here, same reasoning: hard-gating regressed a previously-good result
    # in placement_3d.py's testing, so this stays visualization-only).
    if not args.no_wall_seg:
        wall_mask = segment_wall_mask(rgb, device)
        save_wall_seg_viz(bgr, wall_mask, out_dir / "wall_segmentation.png")
        print(f"Saved {out_dir / 'wall_segmentation.png'} ({wall_mask.mean() * 100:.0f}% of frame, diagnostic only)")

    # ── wall plane: evaluate every well-supported RANSAC line ────────────
    candidates_wf = wall_frame_ransac_multi(pts_cm, up_world, ruck_X, z_floor, cm_per_unit=1.0,
                                            near_cm=args.wall_near_cm)
    if not candidates_wf:
        print("FAILED: no dominant vertical wall found near the Rücklauf")
        return
    print(f"Wall candidates found: {len(candidates_wf)}")
    for i, c in enumerate(candidates_wf):
        print(f"  [{i}] inliers={c['n_inl']}  view_angle={c['view_angle']:.1f}deg  "
              f"origin={np.round(c['origin'], 1).tolist()}")

    def density_filter(pts_uv, radius=6.0, min_neighbors=3):
        if len(pts_uv) < 2:
            return pts_uv
        keep = np.zeros(len(pts_uv), dtype=bool)
        for i in range(0, len(pts_uv), 512):
            chunk = pts_uv[i:i + 512]
            dist = np.linalg.norm(chunk[:, None, :] - pts_uv[None, :, :], axis=2)
            keep[i:i + 512] = (dist < radius).sum(axis=1) >= min_neighbors
        return pts_uv[keep]

    def search_wall(wall, wall_idx):
        """Full obstacle/window/candidate-grid search on ONE wall plane.
        Returns a list of valid candidate dicts (each tagged with which
        wall it came from), or [] if this wall has no usable evidence/space."""
        origin, u_ax, v_ax, n_ax = wall["origin"], wall["u"], wall["v"], wall["n"]
        view_angle, n_inl = wall["view_angle"], wall["n_inl"]
        wall_tilt = float(np.degrees(np.arcsin(abs(n_ax @ up_world))))
        tag = f"wall[{wall_idx}] ({n_inl} inl, {view_angle:.1f}deg off face-on)"
        if wall_tilt > 25.0 or view_angle > 60.0:
            return []

        d_plane = (pts_cm - origin) @ n_ax
        uv = np.stack([(pts_cm - origin) @ u_ax, (pts_cm - origin) @ v_ax], axis=1)
        cam_side = np.sign(float((-origin) @ n_ax))
        d_front = d_plane * cam_side

        wall_band = np.abs(d_plane) < args.wall_band_cm
        obst_band = (d_front >= args.wall_band_cm) & (d_front < 40.0)
        uv_wall = uv[wall_band]
        uv_obst = density_filter(uv[obst_band])

        def fold(masks):
            nonlocal uv_obst
            if not masks:
                return 0
            hit = np.zeros(len(pix), dtype=bool)
            for m in masks:
                mh, mw = m.shape[:2]
                rows = np.clip(pix[:, 0].astype(int), 0, mh - 1)
                cols = np.clip(pix[:, 1].astype(int), 0, mw - 1)
                hit |= m.astype(bool)[rows, cols]
            n_hit = int(hit.sum())
            if n_hit:
                uv_obst = np.concatenate([uv_obst, density_filter(uv[hit])])
            return n_hit

        n_pipe_hit = fold(pipe_masks)
        n_fit_hit = fold(fitting_masks)
        # Real failure on IMG_3126_0089: electrical boxes were only ever
        # used for proximity SCORING (closer = better), never as an
        # obstacle — the top candidate landed squarely on a junction-box
        # cluster because zero distance is literally the best score for
        # that term. "Near electrical" should mean near, not on top of.
        n_elec_hit = fold(elec_masks)

        uv_hole = np.empty((0, 2))
        if window_box is not None:
            x0, y0, x1, y1 = window_box
            in_box = ((pix[:, 1] >= x0) & (pix[:, 1] <= x1) &
                      (pix[:, 0] >= y0) & (pix[:, 0] <= y1))
            hole_band = in_box & (d_front <= -8.0) & (d_front > -120.0)
            uv_hole = density_filter(uv[hole_band])
            if len(uv_hole) == 0:
                uv_hole = density_filter(uv[in_box & wall_band])

        uv_evidence = np.concatenate([uv_wall, uv_obst] + ([uv_hole] if len(uv_hole) else []))
        if len(uv_evidence) < 20:
            print(f"  {tag}: not enough wall evidence, skipping")
            return []
        lo = np.percentile(uv_evidence, 1, axis=0)
        hi = np.percentile(uv_evidence, 99, axis=0)

        origin_z = float((origin @ R_up.T)[2])
        v_floor = z_floor - origin_z
        elec_uv = np.array([[float((p - origin) @ u_ax), float((p - origin) @ v_ax)] for p in elec_pts])

        W, H = args.unit_wh_cm
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
                center_cam = origin + cu * u_ax + cv_ * v_ax
                dist_ruck = float(np.linalg.norm(center_cam - ruck_X))
                dist_elec = float(np.min(np.linalg.norm(elec_uv - [cu, cv_], axis=1))) if len(elec_uv) else 0.0
                mount_h = cv_ - v_floor
                score = (args.w_dist * dist_ruck
                         + args.w_elec * dist_elec
                         + args.w_clear * max(0.0, args.min_clearance_cm - clear)
                         + args.w_height * abs(mount_h - args.preferred_height_cm))
                cands_out.append(dict(u_cm=round(float(cu), 1), v_cm=round(float(cv_), 1),
                                      center_cam=center_cam.tolist(),
                                      dist_to_rucklauf_cm=round(dist_ruck, 1),
                                      dist_to_electrical_cm=round(dist_elec, 1),
                                      clearance_cm=round(clear, 1),
                                      mount_height_cm=round(mount_h, 1),
                                      score=round(score, 1),
                                      wall_idx=wall_idx, wall_view_angle=round(view_angle, 1),
                                      wall_n_inl=n_inl,
                                      wall_origin=origin.tolist(), wall_u=u_ax.tolist(),
                                      wall_v=v_ax.tolist()))
        print(f"  {tag}: {n_cells} cells, {n_hole_rej} window-rejected, {n_obst_rej} obstacle-rejected, "
              f"{len(cands_out)} valid "
              f"(pipe-obst={n_pipe_hit}, fitting-obst={n_fit_hit}, elec-obst={n_elec_hit}, "
              f"wall-pts={len(uv_wall)})")
        return cands_out

    # Search every well-supported wall, not just the most face-on one — a
    # room usually has several real vertical surfaces (front wall, side
    # walls), and free space can exist on any of them. Distance-to-Rücklauf
    # scoring already penalizes far walls naturally; this just stops
    # silently discarding walls that were never even searched.
    #
    # Capping by inlier count ALONE is a real bug, found on IMG_3126_0089:
    # a big, mostly-edge-on wall can have 10x the inliers of a smaller
    # genuinely face-on one (more of the image = more points, regardless of
    # angle), so an inliers-only top-K silently excluded every good-angle
    # candidate (all under 5deg off face-on) in favor of a top-15 entirely
    # made of 85-90deg edge-on near-duplicates — which then all failed the
    # angle gate anyway, wasting the whole search. Union of top-K by
    # inliers AND top-K by face-on-ness so neither a well-evidenced big
    # wall nor a smaller well-oriented one loses its slot to the other.
    by_inliers = sorted(candidates_wf, key=lambda c: -c["n_inl"])[:args.max_walls_searched]
    by_angle = sorted(candidates_wf, key=lambda c: c["view_angle"])[:args.max_walls_searched]
    seen_ids = set()
    search_set = []
    for c in by_angle + by_inliers:   # angle-good candidates tried first
        if id(c) not in seen_ids:
            seen_ids.add(id(c))
            search_set.append(c)
    # A good ANGLE doesn't mean good EVIDENCE. Real failure on IMG_3126_0089:
    # a 1003-inlier wall (good angle, thin point cloud) still put the top
    # candidate on top of a visible electrical cluster — its obstacle
    # points, once projected into that wall's noisy (u,v) frame, didn't
    # land where the objects actually are precisely enough for 5cm-grid
    # collision checks to catch it. Below this floor, a wall's candidates
    # are still computed (for visibility/debugging) but excluded from
    # ranking — the well-evidenced walls found elsewhere are more trustworthy.
    n_thin = sum(1 for c in search_set if c["n_inl"] < args.min_wall_inliers)
    if n_thin:
        print(f"  ({n_thin}/{len(search_set)} candidate walls below --min-wall-inliers "
              f"{args.min_wall_inliers} — searched for visibility, excluded from ranking)")
    print(f"Searching {len(search_set)}/{len(candidates_wf)} wall candidates for free space "
          f"(capped by inlier count):")
    all_cands = []
    for i, wall in enumerate(search_set):
        all_cands.extend(search_wall(wall, i))

    if not all_cands:
        print("FAILED: no valid placement found on any wall candidate")
        return

    trustworthy = [c for c in all_cands if c["wall_n_inl"] >= args.min_wall_inliers]
    if not trustworthy:
        print(f"FAILED: {len(all_cands)} candidate(s) found, but none on a wall with "
              f"--min-wall-inliers {args.min_wall_inliers} — too little evidence to trust "
              f"the obstacle math. Lower the threshold to see them anyway.")
        return
    all_cands = trustworthy

    all_cands.sort(key=lambda c: c["score"])
    top = []
    for c in all_cands:
        if all(c["wall_idx"] != t["wall_idx"] or
               np.hypot(c["u_cm"] - t["u_cm"], c["v_cm"] - t["v_cm"]) > args.unit_wh_cm[0]
               for t in top):
            top.append(c)
        if len(top) == 3:
            break
    for i, c in enumerate(top, 1):
        print(f"  #{i}: wall[{c['wall_idx']}] score={c['score']}  d(Rücklauf)={c['dist_to_rucklauf_cm']}cm  "
              f"d(electrical)={c['dist_to_electrical_cm']}cm  clearance={c['clearance_cm']}cm  "
              f"height={c['mount_height_cm']}cm")

    # ── overlay: project the #1 rectangle's corners back into the photo,
    #    using ITS OWN wall's frame (candidates can come from different walls) ──
    best = top[0]
    origin = np.array(best["wall_origin"])
    u_ax = np.array(best["wall_u"])
    v_ax = np.array(best["wall_v"])
    W, H = args.unit_wh_cm
    corners_uv = [(-W / 2, -H / 2), (W / 2, -H / 2), (W / 2, H / 2), (-W / 2, H / 2)]
    overlay = bgr.copy()
    poly = []
    for du, dv in corners_uv:
        pt_cam = origin + (best["u_cm"] + du) * u_ax + (best["v_cm"] + dv) * v_ax
        px = project_to_pixel(pt_cam, fx, fy, cx, cy)
        if px is None:
            poly = None
            break
        poly.append(px)
    if poly:
        poly = np.array(poly, dtype=np.int32)
        cv2.polylines(overlay, [poly], True, (0, 255, 0), 3)
    cv2.circle(overlay, (int(ruck_px), int(ruck_py)), 8, (255, 0, 0), -1)
    if window_box is not None:
        x0, y0, x1, y1 = window_box.astype(int)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), 2)

    cv2.imwrite(str(out_dir / "overlay.png"), overlay)

    result = dict(
        image=str(args.image), depth_model=args.depth_model, focal_px=fx,
        rucklauf_source=ruck_source, rucklauf_px=[ruck_px, ruck_py],
        n_electricals=len(elec_pts), window_source=window_source,
        window_box=window_box.tolist() if window_box is not None else None,
        n_pipes=len(pipe_boxes) if pipe_boxes is not None else None,
        n_fittings=len(fitting_boxes) if fitting_boxes is not None else None,
        n_wall_candidates_searched=len(candidates_wf),
        unit_wh_cm=[W, H], top_candidates=top,
    )
    (out_dir / "placement.json").write_text(json.dumps(result, indent=2))
    print(f"\nSaved {out_dir / 'placement.json'}")
    print(f"Saved {out_dir / 'overlay.png'}")


if __name__ == "__main__":
    main()
