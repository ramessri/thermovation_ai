"""2D placement MVP: find the best wall position for an indoor-unit footprint.

Reverse-engineers the v3 placement-recommendation stage on a single frame:
the user clicks a scale reference, the Ruecklauf point, and the free-wall
region (none of these are auto-detected yet), obstacles are detected with
the existing GDINO/YOLO-World + SAM2 pipeline, and candidate footprints are
scored by distance to Ruecklauf with a clearance penalty. Assumes the wall
is roughly fronto-parallel to the camera (no perspective correction).
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from experiment_pipeline import (
    load_sam2, load_gdino, load_yolo_world,
    detect_gdino, detect_yolo_world, sam2_from_boxes,
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_calibration"))
from detect_marker import detect_marker

WINDOW = "placement_mvp"


def click_points(image: np.ndarray, instructions: str, n: int | None = None) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    canvas = image.copy()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(WINDOW, canvas)

    print(instructions)
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_click)
    cv2.imshow(WINDOW, canvas)
    while True:
        key = cv2.waitKey(20) & 0xFF
        if n is not None and len(points) >= n:
            break
        if key in (13, 32) and len(points) >= (n or 3):
            break
        if key == 27:
            raise KeyboardInterrupt("point selection aborted")
    cv2.destroyWindow(WINDOW)
    return points


def load_or_pick_points(image: np.ndarray, points_json: Path | None) -> dict:
    if points_json and points_json.exists():
        data = json.loads(points_json.read_text())
        print(f"Loaded calibration points from {points_json}")
        return data

    scale_pts = click_points(
        image, "STEP 1/3: click 2 points of KNOWN real-world distance (e.g. scale-marker edge).", n=2,
    )
    rucklauf_pt = click_points(
        image, "STEP 2/3: click the Ruecklauf (return-flow) connection point.", n=1,
    )[0]
    wall_polygon = click_points(
        image, "STEP 3/3: click >=3 polygon corners around the free wall region, then press SPACE.",
    )

    data = dict(scale_points=scale_pts, rucklauf_point=rucklauf_pt, wall_polygon=wall_polygon)
    if points_json:
        points_json.parent.mkdir(parents=True, exist_ok=True)
        points_json.write_text(json.dumps(data, indent=2))
        print(f"Saved calibration points to {points_json}")
    return data


def build_obstacle_mask(image_rgb: np.ndarray, device: str,
                        gdino_prompt: str, yolo_classes: list[str],
                        threshold: float) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    sam2_pred = load_sam2(device)

    gdino_proc, gdino_model = load_gdino(device)
    boxes, _, _ = detect_gdino(image_rgb, gdino_proc, gdino_model, gdino_prompt, threshold, device)
    for m in sam2_from_boxes(sam2_pred, image_rgb, boxes):
        mask |= (m.squeeze() > 0).astype(np.uint8)

    yolo_model = load_yolo_world(device)
    boxes, _, _ = detect_yolo_world(image_rgb, yolo_model, yolo_classes, threshold)
    for m in sam2_from_boxes(sam2_pred, image_rgb, boxes):
        mask |= (m.squeeze() > 0).astype(np.uint8)

    return mask


def score_candidates(obstacle_mask: np.ndarray, wall_mask: np.ndarray,
                     rucklauf_pt: tuple[int, int], px_per_inch: float,
                     box_w_in: float, box_h_in: float,
                     min_clearance_in: float, stride_px: int,
                     w_dist: float, w_clearance: float) -> list[dict]:
    box_w_px = max(int(round(box_w_in * px_per_inch)), 1)
    box_h_px = max(int(round(box_h_in * px_per_inch)), 1)

    free_mask = ((wall_mask > 0) & (obstacle_mask == 0)).astype(np.uint8)
    dist_to_obstacle = cv2.distanceTransform((1 - obstacle_mask).astype(np.uint8), cv2.DIST_L2, 5)

    ys, xs = np.where(wall_mask > 0)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()

    candidates = []
    for top in range(y0, max(y1 - box_h_px, y0) + 1, stride_px):
        for left in range(x0, max(x1 - box_w_px, x0) + 1, stride_px):
            bottom, right = top + box_h_px, left + box_w_px
            if bottom > wall_mask.shape[0] or right > wall_mask.shape[1]:
                continue
            patch_free = free_mask[top:bottom, left:right]
            if patch_free.mean() < 0.95:
                continue
            clearance_in = float(dist_to_obstacle[top:bottom, left:right].min() / px_per_inch)
            cx, cy = left + box_w_px / 2, top + box_h_px / 2
            dist_to_rucklauf_in = float(np.hypot(cx - rucklauf_pt[0], cy - rucklauf_pt[1]) / px_per_inch)
            clearance_penalty = max(0.0, min_clearance_in - clearance_in)
            score = w_dist * dist_to_rucklauf_in + w_clearance * clearance_penalty
            candidates.append(dict(
                x=left, y=top, w=box_w_px, h=box_h_px,
                dist_to_rucklauf_in=round(dist_to_rucklauf_in, 1),
                clearance_in=round(clearance_in, 1),
                score=round(score, 2),
            ))
    return candidates


def non_overlapping_top_k(candidates: list[dict], k: int, min_sep_px: float) -> list[dict]:
    candidates = sorted(candidates, key=lambda c: c["score"])
    picked: list[dict] = []
    for c in candidates:
        cx, cy = c["x"] + c["w"] / 2, c["y"] + c["h"] / 2
        if all(np.hypot(cx - (p["x"] + p["w"] / 2), cy - (p["y"] + p["h"] / 2)) >= min_sep_px for p in picked):
            picked.append(c)
        if len(picked) >= k:
            break
    return picked


def draw_candidates(image: np.ndarray, rucklauf_pt: tuple[int, int], top_k: list[dict]) -> np.ndarray:
    out = image.copy()
    cv2.drawMarker(out, rucklauf_pt, (0, 0, 255), cv2.MARKER_STAR, 24, 3)
    cv2.putText(out, "Ruecklauf", (rucklauf_pt[0] + 10, rucklauf_pt[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    colors = [(0, 220, 0), (0, 200, 255), (255, 120, 0)]
    for rank, c in enumerate(top_k):
        color = colors[rank % len(colors)]
        cv2.rectangle(out, (c["x"], c["y"]), (c["x"] + c["w"], c["y"] + c["h"]), color, 3)
        label = f"#{rank + 1} score={c['score']} d={c['dist_to_rucklauf_in']}in clr={c['clearance_in']}in"
        cv2.putText(out, label, (c["x"], max(c["y"] - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser(description="2D placement MVP for the Thermovation indoor unit")
    parser.add_argument("image", type=Path)
    parser.add_argument("--box-wh-in", type=float, nargs=2, default=[8.0, 6.0],
                        help="Unit footprint width x height in inches (default: 8 6)")
    parser.add_argument("--ref-distance-in", type=float, default=None,
                        help="Real-world distance in inches between the 2 points you will click for scale "
                             "(required unless --auto-scale is used)")
    parser.add_argument("--gdino-prompt", default="boiler. pipe. valve. pressure gauge.")
    parser.add_argument("--yolo-classes", default="boiler,pipe,valve,pressure gauge")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--min-clearance-in", type=float, default=4.0,
                        help="Desired minimum clearance to nearest obstacle (default: 4in)")
    parser.add_argument("--w-dist", type=float, default=1.0)
    parser.add_argument("--w-clearance", type=float, default=3.0)
    parser.add_argument("--stride-px", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--points-json", type=Path, default=None,
                        help="Reuse/save calibration points instead of re-clicking")
    parser.add_argument("--auto-scale", action="store_true",
                        help="Detect the fiducial marker automatically to set scale "
                             "(skips the two-point click; still prompts for Ruecklauf + wall polygon)")
    parser.add_argument("--output", type=Path, default=Path("output/placement_mvp"))
    args = parser.parse_args()

    if not args.auto_scale and args.ref_distance_in is None:
        parser.error("--ref-distance-in is required unless --auto-scale is used")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise FileNotFoundError(args.image)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = bgr.shape[:2]

    if args.auto_scale:
        px_per_cm, marker_corners, _, _, debug_img = detect_marker(bgr, debug=True)
        if px_per_cm is None:
            raise RuntimeError(
                "Fiducial marker not detected. Check lighting/focus or use manual scale."
            )
        px_per_inch = px_per_cm * 2.54
        print(f"Auto-scale  : {px_per_cm:.3f} px/cm  →  {px_per_inch:.2f} px/inch")
        if debug_img is not None:
            marker_out = args.output / f"{args.image.stem}_marker.jpg"
            args.output.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(marker_out), debug_img)
            print(f"Marker debug: {marker_out}")
        rucklauf_pt = tuple(
            click_points(bgr, "Click the Ruecklauf (return-flow) connection point.", n=1)[0]
        )
        polygon = np.array(
            click_points(bgr, "Click >=3 polygon corners around the free wall region, then SPACE."),
            dtype=np.int32,
        )
    else:
        points = load_or_pick_points(bgr, args.points_json)
        (sx1, sy1), (sx2, sy2) = points["scale_points"]
        px_per_inch = np.hypot(sx2 - sx1, sy2 - sy1) / args.ref_distance_in
        rucklauf_pt = tuple(points["rucklauf_point"])
        polygon = np.array(points["wall_polygon"], dtype=np.int32)
    print(f"Scale: {px_per_inch:.2f} px/inch")

    wall_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(wall_mask, [polygon], 1)

    yolo_classes = [c.strip() for c in args.yolo_classes.split(",")]
    print("Detecting obstacles...")
    obstacle_mask = build_obstacle_mask(rgb, device, args.gdino_prompt, yolo_classes, args.threshold)

    print("Scoring candidate placements...")
    candidates = score_candidates(
        obstacle_mask, wall_mask, rucklauf_pt, px_per_inch,
        args.box_wh_in[0], args.box_wh_in[1], args.min_clearance_in,
        args.stride_px, args.w_dist, args.w_clearance,
    )
    if not candidates:
        print("No valid placement found: free-wall region too small/occluded for this box size.")
        return

    box_w_px = max(int(round(args.box_wh_in[0] * px_per_inch)), 1)
    top_k = non_overlapping_top_k(candidates, args.top_k, min_sep_px=box_w_px)

    out_img = draw_candidates(bgr, rucklauf_pt, top_k)
    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / f"{args.image.stem}_placement.jpg"
    cv2.imwrite(str(out_path), out_img)

    summary_path = args.output / f"{args.image.stem}_placement.json"
    summary_path.write_text(json.dumps(dict(
        px_per_inch=px_per_inch, rucklauf_point=rucklauf_pt,
        box_wh_in=args.box_wh_in, candidates_evaluated=len(candidates),
        top_k=top_k,
    ), indent=2))

    print(f"Saved visualization to {out_path}")
    print(f"Saved summary to {summary_path}")
    for rank, c in enumerate(top_k, 1):
        print(f"  #{rank}: score={c['score']}  dist_to_rucklauf={c['dist_to_rucklauf_in']}in  clearance={c['clearance_in']}in")


if __name__ == "__main__":
    main()
