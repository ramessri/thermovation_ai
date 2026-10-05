"""
Qwen2-VL-2B as the obstacle detector in the full placement pipeline: same
rectification, occupancy grid, clearance inflation, search and overlay as
wall_rectify.py — only the source of obstacle boxes changes (one
single-object query per term, box masks, no SAM2). No manual obstacles and
no cross-wall unfold, so every wall is scored on its own.

This is the test that reversed qwen_pipe_detection.py's promising isolated
result: run-to-run inconsistency (a pipe found in isolation was missed here,
and a candidate landed on a pipe elbow), occasional wall-sized hallucinated
boxes, and the same fixed-vocabulary blind spots as GDINO (README §5.1).

Run: python 2d_wall_rectify/qwen_obstacle_placement.py
"""

from __future__ import annotations

import json

from config import IMAGES_DIR, OUTPUT_DIR, load_walls
from vlm import QWEN2_2B, load_pil, load_qwen, query_single_object
from wall_rectify import (box_mask, describe, distance_score_fn, iou, load_image, locate_rucklauf, place,
                          rectify, render_overlay, warp_mask)

OUT_DIR = OUTPUT_DIR / "qwen_obstacle_placement"

# GDINO's vocabulary minus "heating pipe" (returned the same box as "pipe")
# and "thermometer" (never found by any detector here), plus two terms for
# things visible in these photos.
OBSTACLE_QUERIES = [
    "boiler", "hot water tank", "pipe", "gas hose", "valve manifold",
    "pump", "electrical box", "window", "door", "radiator",
]

# Candidates found by wall_rectify.py with grounding-dino-base, for comparison.
GDINO_CANDIDATES = {"wall1": 3, "wall2": 3, "wall3": 0, "wall4": 3}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, processor = load_qwen(QWEN2_2B)

    results = {}
    for wall in load_walls():
        print(f"\n=== {wall.name} ({wall.file}) — {wall.width_cm:.0f}x{wall.height_cm:.0f}cm ===")
        img = load_image(IMAGES_DIR / wall.file)
        img_h, img_w = img.shape[:2]
        pil_img = load_pil(IMAGES_DIR / wall.file)
        rect = rectify(wall, img)

        detections: list[dict] = []
        for obj in OBSTACLE_QUERIES:
            box, _ = query_single_object(model, processor, pil_img, obj, img_w, img_h)
            # Different query terms often land on the same object ("pipe" / "gas hose").
            if box is not None and not any(iou(box, d["box"]) > 0.6 for d in detections):
                detections.append({"label": obj, "box": box})
        print(f"  Qwen2-VL: {len(detections)} obstacle(s): " +
              (", ".join(d["label"] for d in detections) or "none"))
        for d in detections:
            d["warped_mask"] = warp_mask(rect.H, box_mask(d["box"], (img_h, img_w)), rect.width_px, rect.height_px)

        rucklauf_px, rucklauf_note = locate_rucklauf(wall, img, rect.H)
        print(f"  Rücklauf: {rucklauf_note}")
        score_fn = distance_score_fn(rucklauf_px) if rucklauf_px is not None else None

        occupied, candidates = place(rect, detections, score_fn)
        render_overlay(rect.image, occupied, candidates, rucklauf_px, OUT_DIR / f"{wall.name}_overlay.png")
        print(f"  {len(candidates)} candidate(s)")
        for rank, c in enumerate(candidates, start=1):
            print(f"    #{rank}  {describe(c)}")

        results[wall.name] = {
            "obstacles": [{"label": d["label"], "box": d["box"]} for d in detections],
            "candidates": [{k: v for k, v in c.items() if k != "score"} for c in candidates],
        }

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("\n" + "=" * 70)
    print("Candidates: grounding-dino-base vs Qwen2-VL-2B as the obstacle detector")
    for wall_name, r in results.items():
        print(f"  {wall_name}: GDINO={GDINO_CANDIDATES[wall_name]}  "
              f"Qwen2-VL={len(r['candidates'])} ({len(r['obstacles'])} obstacles)")
    print(f"\nOutputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
