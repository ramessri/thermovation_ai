"""
Lightweight alternative to the 3D placement chain (06_placement/placement_3d.py,
which needs a full SfM+MVS video scan): rectify ONE still photo per wall to
metric scale with a plain 2D homography (4 clicked corners + the wall's
known width/height), then run obstacle detection, Rücklauf localization and
mounting-spot search directly in that flat image — no point cloud.

Per wall:
  1. Homography from the 4 corners (corners.json) to a PX_PER_CM canvas.
  2. Scale sanity checks: ArUco markers (15cm) and the boiler/tank (known size).
  3. Obstacles: GroundingDINO boxes refined to SAM2 masks, plus manual boxes.
  4. Rücklauf: blue cap/dial via HSV, preferring one with a red blob nearby.
  5. Occupancy grid with per-class clearance inflation, grid-search the unit
     footprint, score by distance to the Rücklauf (else max clearance).
  6. Top-3 candidates as an overlay image + results.json.

The functions here are also imported by the comparison scripts in this folder.

Usage:
    python 2d_wall_rectify/wall_rectify.py
    python 2d_wall_rectify/wall_rectify.py --gdino-model openmmlab-community/mm_grounding_dino_large_all
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from config import (ARUCO_MARKER_SIZE_CM, KNOWN_BOILER_H_CM, KNOWN_BOILER_W_CM, KNOWN_TANK_H_CM,
                    IMAGES_DIR, LOW_CONFIDENCE_WALLS, MANUAL_OBSTACLE_BOXES, MANUAL_RUCKLAUF_PX,
                    OUTPUT_DIR, ROOT, WallSpec, load_walls)

PX_PER_CM = 4.0

DEFAULT_GDINO_MODEL = "IDEA-Research/grounding-dino-base"
# "heating pipe" as its own prompt (not just "pipe") changes what GDINO
# surfaces — it recovered a missed hot water tank on wall3. 0.20 surfaces real
# extra pipe/fixture boxes on all 4 walls; empty-label junk starts below ~0.18.
GDINO_PROMPTS = [
    "boiler", "hot water tank", "pipe", "heating pipe", "valve", "pump",
    "thermometer", "electrical box", "window", "door", "radiator",
]
BOX_THRESHOLD = 0.20
GDINO_MAX_DIM = 1400   # detection downscale; full-res gave identical detections

# Thermovation indoor unit footprint on the wall (Länge x Höhe). Its 31cm
# depth (Tiefe) can't be modelled — the 2D grid has no notion of standoff.
UNIT_W_CM, UNIT_H_CM = 31.0, 48.5

BASE_CLEARANCE_CM = 8.0   # 5-10cm service clearance
# Extra inflation for classes that stand proud of the wall: a pipe 10cm in
# front of a spot still looks "free" in a flat photo. Cheap mitigation, not a
# depth check.
EXTRA_CLEARANCE_CM = {
    "pipe": 12.0, "valve": 12.0, "pump": 20.0,
    "boiler": 20.0, "hot water tank": 20.0, "radiator": 8.0, "manifold": 12.0,
}
NMS_MIN_SEP_CM = max(UNIT_W_CM, UNIT_H_CM)

# Rücklauf (blue) / Vorlauf (red) cap colours. S/V floors are 35, not the
# usual 80/50: the manifold dials in this dim basement sample at S~50-57,
# V~44-62. Re-sample real fixture pixels before reusing on another dataset.
BLUE_HSV = (np.array([95, 35, 35]), np.array([135, 255, 255]))
RED_HSV_LO = (np.array([0, 35, 35]), np.array([15, 255, 255]))
RED_HSV_HI = (np.array([155, 35, 35]), np.array([179, 255, 255]))
MIN_BLOB_AREA_PX = 20
MAX_BLOB_AREA_PX = 3000
MIN_CIRCULARITY = 0.55
RUCKLAUF_SCAN_MAX_DIM = 1920       # blob-area thresholds above are tuned at this scale
RUCKLAUF_PAIR_MAX_DIST_PX = 120    # "blue/red pair next to each other", ~15cm dial spacing at this scale


# ── Geometry ────────────────────────────────────────────────────────────────

@dataclass
class RectifiedWall:
    H: np.ndarray            # original image px -> canvas px
    width_px: int
    height_px: int
    image: np.ndarray        # rectified BGR canvas
    valid_mask: np.ndarray   # canvas pixels actually covered by the photo


def build_homography(corners_px, width_cm: float, height_cm: float,
                     px_per_cm: float = PX_PER_CM) -> tuple[np.ndarray, int, int]:
    src = np.array(corners_px, dtype=np.float32)
    w_px, h_px = width_cm * px_per_cm, height_cm * px_per_cm
    dst = np.array([[0, 0], [w_px, 0], [w_px, h_px], [0, h_px]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst), int(round(w_px)), int(round(h_px))


def warp_point(H: np.ndarray, pt) -> tuple[float, float]:
    q = H @ np.array([pt[0], pt[1], 1.0])
    return float(q[0] / q[2]), float(q[1] / q[2])


def warp_mask(H: np.ndarray, mask: np.ndarray, canvas_w: int, canvas_h: int) -> np.ndarray:
    warped = cv2.warpPerspective(mask.astype(np.uint8) * 255, H, (canvas_w, canvas_h), flags=cv2.INTER_NEAREST)
    return warped > 127


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    return img


def rectify(wall: WallSpec, img: np.ndarray) -> RectifiedWall:
    H, cw, ch = build_homography(wall.corners_px, wall.width_cm, wall.height_cm)
    image = cv2.warpPerspective(img, H, (cw, ch), flags=cv2.INTER_LINEAR, borderValue=(20, 20, 20))
    valid_mask = cv2.warpPerspective(np.full(img.shape[:2], 255, np.uint8), H, (cw, ch),
                                     flags=cv2.INTER_NEAREST) > 0
    return RectifiedWall(H, cw, ch, image, valid_mask)


# ── Scale sanity checks ─────────────────────────────────────────────────────

def detect_aruco(img_bgr: np.ndarray) -> list[dict]:
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100),
                                       cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(img_bgr)
    if ids is None:
        return []
    return [{"id": int(i), "corners_px": c[0].tolist()} for c, i in zip(corners, ids.flatten())]


def marker_sanity_checks(H: np.ndarray, markers: list[dict]) -> list[dict]:
    """
    Warp each marker's corners through this wall's H and compare the side
    length to the known marker size. A marker flush on the wall warps to a
    near-square; one on another plane (the table in wall1's photo, a side
    wall) warps to a skewed quad — squareness excludes those automatically.
    """
    checks = []
    for mk in markers:
        warped = np.array([warp_point(H, p) for p in mk["corners_px"]])
        sides = np.linalg.norm(warped - np.roll(warped, -1, axis=0), axis=1) / PX_PER_CM
        diag1 = np.linalg.norm(warped[0] - warped[2])
        diag2 = np.linalg.norm(warped[1] - warped[3])
        squareness = min(diag1, diag2) / max(diag1, diag2) if max(diag1, diag2) > 0 else 0.0
        avg_side_cm = float(np.mean(sides))
        err_pct = abs(avg_side_cm - ARUCO_MARKER_SIZE_CM) / ARUCO_MARKER_SIZE_CM * 100.0
        checks.append({
            "id": mk["id"], "avg_side_cm": round(avg_side_cm, 1),
            "err_pct": round(err_pct, 1), "squareness": round(float(squareness), 2),
            "on_plane": bool(squareness > 0.85 and 5.0 < avg_side_cm < 40.0),
        })
    return checks


def reference_object_checks(detections: list[dict]) -> list[str]:
    """Rectified size of the boiler/tank vs. their known dimensions. They stand
    proud of the wall, so this also measures the parallax error of anything
    not flush with the wall plane. Manual boxes are skipped — they are drawn
    loosely as keep-outs, not as measurements."""
    msgs = []
    for det in detections:
        if det.get("manual"):
            continue
        label = det["label"].lower()
        ys, xs = np.where(det["warped_mask"])
        if len(xs) == 0:
            continue
        w_cm = (xs.max() - xs.min()) / PX_PER_CM
        h_cm = (ys.max() - ys.min()) / PX_PER_CM
        if "boiler" in label:
            werr = abs(w_cm - KNOWN_BOILER_W_CM) / KNOWN_BOILER_W_CM * 100
            herr = abs(h_cm - KNOWN_BOILER_H_CM) / KNOWN_BOILER_H_CM * 100
            tag = "PASS" if max(werr, herr) < 5 else "WARN"
            msgs.append(f"boiler '{det['label']}' measured {w_cm:.0f}x{h_cm:.0f}cm "
                        f"(known {KNOWN_BOILER_W_CM:.0f}x{KNOWN_BOILER_H_CM:.0f}cm) — "
                        f"W err {werr:.0f}%, H err {herr:.0f}% — {tag}")
        elif "tank" in label:
            herr = abs(h_cm - KNOWN_TANK_H_CM) / KNOWN_TANK_H_CM * 100
            tag = "PASS" if herr < 5 else "WARN"
            msgs.append(f"tank '{det['label']}' measured height {h_cm:.0f}cm "
                        f"(known {KNOWN_TANK_H_CM:.0f}cm) — err {herr:.0f}% — {tag}")
    return msgs


# ── Obstacle detection (GroundingDINO + SAM2) ───────────────────────────────

class ObstacleDetector:
    """GroundingDINO boxes refined to SAM2 masks. Any checkpoint loadable via
    AutoModelForZeroShotObjectDetection works (grounding-dino-base,
    mm_grounding_dino_large_all, ...). If SAM2 fails to load, boxes are used
    as filled-rectangle masks instead."""

    def __init__(self, model_id: str = DEFAULT_GDINO_MODEL, device: Optional[str] = None):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()

        sys.path.insert(0, str(ROOT / "experiments"))
        from experiment_pipeline import load_sam2
        try:
            self.sam2 = load_sam2(self.device)
        except Exception as e:
            print(f"  [warn] SAM2 unavailable ({e}) — using box masks")
            self.sam2 = None

    def detect_boxes(self, image_rgb: np.ndarray, prompts: list[str], threshold: float) -> list[dict]:
        import torch
        from PIL import Image

        pil_img = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil_img, text=". ".join(prompts) + ".", return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        post = dict(text_threshold=threshold, target_sizes=[pil_img.size[::-1]])
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=threshold, **post)[0]
        except TypeError:   # transformers < 4.51 named it box_threshold
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=threshold, **post)[0]
        labels = results["text_labels"] if "text_labels" in results else results["labels"]
        return [{"box": box.tolist(), "label": label, "score": float(score)}
                for box, label, score in zip(results["boxes"], labels, results["scores"])]

    def detect(self, img_bgr: np.ndarray, prompts: list[str] = GDINO_PROMPTS,
               threshold: float = BOX_THRESHOLD) -> list[dict]:
        """Detections in original-image px: {label, score, box, mask}."""
        h, w = img_bgr.shape[:2]
        scale = min(1.0, GDINO_MAX_DIM / max(h, w))
        small = cv2.resize(img_bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else img_bgr
        small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        dets = [d for d in self.detect_boxes(small_rgb, prompts, threshold) if d["label"].strip()]
        if self.sam2 is not None and dets:
            from experiment_pipeline import sam2_from_boxes
            masks_small = sam2_from_boxes(self.sam2, small_rgb, np.array([d["box"] for d in dets]))
        else:
            masks_small = [None] * len(dets)

        results = []
        for d, mask_small in zip(dets, masks_small):
            box = [v / scale for v in d["box"]]
            if mask_small is not None:
                mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST) > 0
            else:
                mask = box_mask(box, (h, w))
            results.append({"label": d["label"], "score": d["score"], "box": box, "mask": mask})
        return results


def box_mask(box, shape: tuple[int, int]) -> np.ndarray:
    x0, y0, x1, y1 = (int(v) for v in box)
    mask = np.zeros(shape, dtype=bool)
    mask[max(0, y0):y1, max(0, x0):x1] = True
    return mask


def manual_obstacles(wall_name: str, shape: tuple[int, int]) -> list[dict]:
    return [{"label": m["label"], "score": 1.0, "box": m["box"], "mask": box_mask(m["box"], shape), "manual": True}
            for m in MANUAL_OBSTACLE_BOXES.get(wall_name, [])]


def canonical_label(label: str) -> str:
    label_l = label.lower()
    for key in ("pipe", "tank", "boiler", "valve", "pump", "manifold",
                "thermometer", "electrical", "window", "door", "radiator"):
        if key in label_l:
            return key
    return label_l


def iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def dedup_detections(detections: list[dict], iou_thresh: float = 0.4) -> list[dict]:
    """At BOX_THRESHOLD=0.20 GDINO returns several overlapping boxes per object
    (one 'tank' measured 14cm tall against a known 181cm). Keep the
    highest-scoring box per canonical label among overlapping ones; manual
    boxes (score 1.0) always win."""
    kept: list[dict] = []
    for d in sorted(detections, key=lambda d: -d["score"]):
        canon = canonical_label(d["label"])
        if not any(canonical_label(k["label"]) == canon and iou(d["box"], k["box"]) > iou_thresh for k in kept):
            kept.append(d)
    return kept


# ── Rücklauf localization (HSV cap colour) ──────────────────────────────────

def find_color_blobs(img_bgr: np.ndarray, ranges: list[tuple]) -> list[tuple[float, float, float]]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, lo, hi)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_BLOB_AREA_PX <= area <= MAX_BLOB_AREA_PX):
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        if r > 0 and area / (np.pi * r * r) >= MIN_CIRCULARITY:
            blobs.append((float(cx), float(cy), float(r)))
    return blobs


def find_rucklauf(img_bgr: np.ndarray, H: np.ndarray) -> tuple[Optional[tuple[float, float]], bool]:
    """Blue cap/dial = Rücklauf, mapped into canvas px. A blue blob with a red
    blob nearby (the Vorlauf/Rücklauf pair) beats a larger lone blue blob,
    which is more likely an incidental blue surface. Returns (position, paired)."""
    h, w = img_bgr.shape[:2]
    scale = min(1.0, RUCKLAUF_SCAN_MAX_DIM / max(h, w))
    small = cv2.resize(img_bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else img_bgr
    blue = find_color_blobs(small, [BLUE_HSV])
    red = find_color_blobs(small, [RED_HSV_LO, RED_HSV_HI])
    if not blue:
        return None, False
    paired = [b for b in blue
              if any(np.hypot(b[0] - r[0], b[1] - r[1]) < RUCKLAUF_PAIR_MAX_DIST_PX for r in red)]
    cx, cy, _ = max(paired or blue, key=lambda b: b[2])
    return warp_point(H, (cx / scale, cy / scale)), bool(paired)


def locate_rucklauf(wall: WallSpec, img_bgr: np.ndarray, H: np.ndarray) -> tuple[Optional[tuple[float, float]], str]:
    """Manual override if one exists for this wall (used instead of, never
    averaged with, the HSV result), else the HSV detector. Returns (canvas
    position or None, human-readable note)."""
    auto_px, paired = find_rucklauf(img_bgr, H)
    if wall.name in MANUAL_RUCKLAUF_PX:
        note = "manual override"
        if auto_px is not None:
            note += f" (HSV detector found {'a paired' if paired else 'an unpaired'} blue blob at canvas " \
                    f"{tuple(round(v) for v in auto_px)} — not used)"
        return warp_point(H, MANUAL_RUCKLAUF_PX[wall.name]), note
    if auto_px is None:
        return None, "not found (no blue colour cue)"
    conf = "paired with a red blob" if paired else "unpaired — low confidence, may be an incidental blue surface"
    return auto_px, f"HSV, {conf}"


# ── Occupancy grid + placement search ───────────────────────────────────────

def label_extra_clearance_cm(label: str) -> float:
    label_l = label.lower()
    return max((v for k, v in EXTRA_CLEARANCE_CM.items() if k in label_l), default=0.0)


def build_occupancy(canvas_w: int, canvas_h: int, detections: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (inflated occupancy mask, distance to nearest raw obstacle in cm).
    Each detection needs a 'warped_mask' in canvas px."""
    occupied = np.zeros((canvas_h, canvas_w), dtype=bool)
    raw_union = np.zeros((canvas_h, canvas_w), dtype=bool)
    for det in detections:
        raw_union |= det["warped_mask"]
        k = max(1, int(round((BASE_CLEARANCE_CM + label_extra_clearance_cm(det["label"])) * PX_PER_CM)))
        occupied |= cv2.dilate(det["warped_mask"].astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
    dist_px = cv2.distanceTransform((~raw_union).astype(np.uint8), cv2.DIST_L2, 5)
    # With no obstacles distanceTransform returns a ~1e37 sentinel; cap at the canvas diagonal.
    dist_px = np.clip(dist_px, 0, float(np.hypot(canvas_w, canvas_h)))
    return occupied, dist_px / PX_PER_CM


def _rect_sum(integral: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> int:
    return int(integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0])


def search_wall(canvas_w: int, canvas_h: int, free_mask: np.ndarray, dist_cm: np.ndarray,
                score_fn: Optional[Callable[[float, float], float]], top_k: int = 3) -> list[dict]:
    """Grid-search unit footprints that lie fully on free, photographed wall.
    Lower score is better: distance to the Rücklauf if score_fn is given,
    else negative clearance. Picks are kept at least NMS_MIN_SEP_CM apart."""
    unit_w_px = int(round(UNIT_W_CM * PX_PER_CM))
    unit_h_px = int(round(UNIT_H_CM * PX_PER_CM))
    if unit_w_px >= canvas_w or unit_h_px >= canvas_h:
        return []
    step_px = max(1, int(round(0.5 * min(unit_w_px, unit_h_px))))
    free_int = cv2.integral(free_mask.astype(np.uint8))
    area = unit_w_px * unit_h_px

    candidates = []
    for y0 in range(0, canvas_h - unit_h_px, step_px):
        for x0 in range(0, canvas_w - unit_w_px, step_px):
            if _rect_sum(free_int, x0, y0, x0 + unit_w_px, y0 + unit_h_px) < area:
                continue
            cx, cy = x0 + unit_w_px / 2.0, y0 + unit_h_px / 2.0
            clearance_cm = float(dist_cm[int(cy), int(cx)])
            if clearance_cm < BASE_CLEARANCE_CM:
                continue
            score = score_fn(cx, cy) if score_fn is not None else -clearance_cm
            candidates.append({"x0": x0, "y0": y0, "cx": cx, "cy": cy,
                               "clearance_cm": round(clearance_cm, 1), "score": score})

    candidates.sort(key=lambda c: c["score"])
    picked: list[dict] = []
    for c in candidates:
        if all(np.hypot(c["cx"] - p["cx"], c["cy"] - p["cy"]) / PX_PER_CM >= NMS_MIN_SEP_CM for p in picked):
            picked.append(c)
            if len(picked) >= top_k:
                break
    return picked


def distance_score_fn(target_px: tuple[float, float]) -> Callable[[float, float], float]:
    rx, ry = target_px
    return lambda cx, cy: float(np.hypot(cx - rx, cy - ry)) / PX_PER_CM


def edge_distance_cm(x_px: float, canvas_w: int, shared_edge: str) -> float:
    return x_px / PX_PER_CM if shared_edge == "left" else (canvas_w - x_px) / PX_PER_CM


def place(rect: RectifiedWall, detections: list[dict],
          score_fn: Optional[Callable[[float, float], float]]) -> tuple[np.ndarray, list[dict]]:
    """Occupancy + search + human-readable distances. Detections need 'warped_mask'.
    Returns (inflated occupancy mask, top candidates)."""
    occupied, dist_cm = build_occupancy(rect.width_px, rect.height_px, detections)
    candidates = search_wall(rect.width_px, rect.height_px, rect.valid_mask & ~occupied, dist_cm, score_fn)
    for c in candidates:
        c["from_floor_cm"] = round((rect.height_px - c["cy"]) / PX_PER_CM, 1)
        c["from_left_corner_cm"] = round(c["x0"] / PX_PER_CM, 1)
        if score_fn is not None:
            c["from_rucklauf_cm"] = round(c["score"], 1)
    return occupied, candidates


def describe(c: dict) -> str:
    ruck = f"{c['from_rucklauf_cm']} cm from Rücklauf" if "from_rucklauf_cm" in c else "no Rücklauf reference"
    return (f"{c['from_floor_cm']} cm from floor, {c['from_left_corner_cm']} cm from left corner, "
            f"{ruck}, clearance {c['clearance_cm']} cm")


# ── Overlay rendering ───────────────────────────────────────────────────────

def render_overlay(rectified: np.ndarray, occupied: np.ndarray, candidates: list[dict],
                   rucklauf_px: Optional[tuple[float, float]], out_path: Path) -> None:
    """Occupancy tinted red, Rücklauf circled, candidates boxed (#1 green,
    others orange) with their distances printed under (or above) each box.
    OpenCV's Hershey font has no umlauts, hence 'Rucklauf'."""
    alpha = 0.35
    tinted = (rectified * (1 - alpha) + np.array([0, 0, 255]) * alpha).astype(np.uint8)
    vis = np.where(occupied[..., None], tinted, rectified)

    if rucklauf_px is not None:
        p = (int(rucklauf_px[0]), int(rucklauf_px[1]))
        cv2.circle(vis, p, 14, (255, 200, 0), 3)
        cv2.putText(vis, "Rucklauf", (p[0] + 16, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

    unit_w_px = int(round(UNIT_W_CM * PX_PER_CM))
    unit_h_px = int(round(UNIT_H_CM * PX_PER_CM))
    canvas_h, canvas_w = vis.shape[:2]
    for rank, c in enumerate(candidates, start=1):
        color = (0, 220, 0) if rank == 1 else (0, 165, 255)
        x0, y0 = int(c["x0"]), int(c["y0"])
        x1, y1 = x0 + unit_w_px, y0 + unit_h_px
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 4 if rank == 1 else 2)

        lines = [
            f"#{rank}",
            f"{c['from_floor_cm']:.0f}cm from floor",
            f"{c['from_left_corner_cm']:.0f}cm from left corner",
            f"{c['from_rucklauf_cm']:.0f}cm from Rucklauf" if "from_rucklauf_cm" in c
            else f"clearance {c['clearance_cm']:.0f}cm",
        ]
        line_h = 17
        block_w, block_h = 165, line_h * len(lines) + 6
        text_x = int(np.clip(x0, 2, canvas_w - block_w - 2))
        text_y = y1 + 6
        if text_y + block_h > canvas_h:   # flip above the box if it would run off the bottom
            text_y = max(2, y0 - block_h - 6)
        cv2.rectangle(vis, (text_x, text_y), (text_x + block_w, text_y + block_h), (0, 0, 0), -1)
        cv2.rectangle(vis, (text_x, text_y), (text_x + block_w, text_y + block_h), color, 1)
        for i, line in enumerate(lines):
            cv2.putText(vis, line, (text_x + 5, text_y + 15 + i * line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), vis)


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gdino-model", default=DEFAULT_GDINO_MODEL,
                        help=f"zero-shot detector checkpoint (default: {DEFAULT_GDINO_MODEL})")
    args = parser.parse_args()

    walls = load_walls()
    out_dir = OUTPUT_DIR / args.gdino_model.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.gdino_model} + SAM2...")
    detector = ObstacleDetector(args.gdino_model)

    # Pass 1: per wall — rectify, sanity-check, detect, locate the Rücklauf.
    wall_data: dict[str, dict] = {}
    for wall in walls:
        print(f"\n=== {wall.name} ({wall.file}) — {wall.width_cm:.0f}x{wall.height_cm:.0f}cm ===")
        img = load_image(IMAGES_DIR / wall.file)
        rect = rectify(wall, img)
        cv2.imwrite(str(out_dir / f"{wall.name}_rectified.jpg"), rect.image)
        if wall.name in LOW_CONFIDENCE_WALLS:
            print(f"  [low confidence] {LOW_CONFIDENCE_WALLS[wall.name]}")

        marker_checks = marker_sanity_checks(rect.H, detect_aruco(img))
        for mc in marker_checks:
            tag = ("PASS" if mc["err_pct"] < 5 else "WARN") if mc["on_plane"] else "excluded (off-plane)"
            print(f"  marker #{mc['id']}: measured {mc['avg_side_cm']}cm "
                  f"(want {ARUCO_MARKER_SIZE_CM:.0f}cm, {mc['err_pct']}% err) — {tag}")

        detections = detector.detect(img)
        print(f"  detector: {len(detections)} detection(s)" +
              "".join(f"\n    {d['label']} ({d['score']:.2f})" for d in detections))
        manual = manual_obstacles(wall.name, img.shape[:2])
        for m in manual:
            print(f"  [manual obstacle] {m['label']}")
        n_before = len(detections) + len(manual)
        detections = dedup_detections(detections + manual)
        if len(detections) < n_before:
            print(f"  [dedup] {n_before} -> {len(detections)} detections after overlap suppression")
        for d in detections:
            d["warped_mask"] = warp_mask(rect.H, d["mask"], rect.width_px, rect.height_px)
        for msg in reference_object_checks(detections):
            print(f"  [scale check] {msg}")

        rucklauf_px, rucklauf_note = locate_rucklauf(wall, img, rect.H)
        print(f"  Rücklauf: {rucklauf_note}")
        wall_data[wall.name] = {"wall": wall, "rect": rect, "detections": detections,
                                "marker_checks": marker_checks, "rucklauf_px": rucklauf_px,
                                "rucklauf_note": rucklauf_note}

    # Pass 2: score + search. A wall with no Rücklauf of its own borrows its
    # adjacent wall's by "unfolding" across the shared corner: distance = own
    # distance to the shared edge + the Rücklauf's distance to that edge.
    print("\n" + "=" * 60)
    results: dict[str, dict] = {}
    for wall in walls:
        d = wall_data[wall.name]
        rect: RectifiedWall = d["rect"]
        score_fn, source = None, None
        if d["rucklauf_px"] is not None:
            score_fn, source = distance_score_fn(d["rucklauf_px"]), "this wall"
        elif wall.adjacent_to and wall_data[wall.adjacent_to]["rucklauf_px"] is not None:
            adj = wall_data[wall.adjacent_to]
            adj_term = edge_distance_cm(adj["rucklauf_px"][0], adj["rect"].width_px, adj["wall"].shared_edge)
            score_fn = lambda cx, cy, w=rect.width_px, e=wall.shared_edge, a=adj_term: \
                edge_distance_cm(cx, w, e) + a  # noqa: E731
            source = f"unfolded via {wall.adjacent_to} (low confidence)"

        occupied, candidates = place(rect, d["detections"], score_fn)
        render_overlay(rect.image, occupied, candidates, d["rucklauf_px"], out_dir / f"{wall.name}_overlay.png")

        print(f"\n{wall.name} ({wall.file}):")
        print(f"  Rücklauf reference: {source or 'none (scored by max clearance)'}")
        if not candidates:
            print("  No valid mounting spot found (wall too small, fully occupied, or out of frame).")
        for rank, c in enumerate(candidates, start=1):
            print(f"  #{rank}  {describe(c)}")

        results[wall.name] = {
            "file": wall.file, "width_cm": wall.width_cm, "height_cm": wall.height_cm,
            "low_confidence_note": LOW_CONFIDENCE_WALLS.get(wall.name),
            "marker_checks": d["marker_checks"],
            "detections": [{"label": det["label"], "score": det["score"]} for det in d["detections"]],
            "rucklauf": d["rucklauf_note"],
            "rucklauf_reference": source,
            "candidates": [{k: v for k, v in c.items() if k != "score"} for c in candidates],
        }

    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nOutputs written to {out_dir}")


if __name__ == "__main__":
    main()
