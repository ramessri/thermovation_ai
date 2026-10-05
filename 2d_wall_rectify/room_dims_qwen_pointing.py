"""
Follow-up to room_dims_qwen_text.py: stop asking the VLM to do arithmetic.
Qwen2.5-VL-7B only returns pixel coordinates of named landmarks; every
scale/distance calculation happens in Python. Walls 3 and 4 (the ones with
the tank in frame). Three variants, in the order tried:

  with-sizes   The landmark schema also states each object's height in cm.
               -> contaminated: the boiler's base->top delta came back as
                  ~129px and the tank's as ~180px — the cm numbers echoed as
                  pixels. The two objects' cm/px scales disagreed by 124%.
  decoupled    The same landmarks with no cm value or size wording anywhere;
               known heights are attached in Python afterwards. Scale is a
               flat average, so far-away corners/ceiling get no perspective
               correction (25-40% error on wall3).
  homography   Adds the wall's 4 corners and solves the wall height with
               room_dims_solver.py (perspective-correct). Failed because
               Qwen's "corners" were just the image frame edges.

Every run draws the returned points onto the photo — check them before
trusting any number derived from them.

Usage: python 2d_wall_rectify/room_dims_qwen_pointing.py --variant decoupled
"""

from __future__ import annotations

import argparse
import json
from typing import Optional

from PIL import Image, ImageDraw

from config import IMAGES_DIR, KNOWN_BOILER_H_CM, KNOWN_TANK_H_CM, OUTPUT_DIR, ROOM_HEIGHT_CM, WALLS
from room_dims_solver import error_report, solve_wall_height
from vlm import QWEN25_7B, ask, load_pil, load_qwen, parse_json_object

OUT_DIR = OUTPUT_DIR / "room_dims_qwen_pointing"

_HEADER = """You are annotating a photo for a computer vision program. Do NOT estimate
any distances, sizes, or measurements of any kind. Only locate objects and
return pixel coordinates [x, y] in this image's resolution ({W} x {H}).
If a point is not visible, return null.
"""

WITH_SIZES_PROMPT = """You are annotating a photo for a measurement program. Do NOT estimate any
distances, sizes or heights. Only return pixel coordinates [x, y] in the
ORIGINAL image resolution ({W} x {H}). If a point is hidden, return null.

Return exactly this JSON:
{{
 "references": [
   {{"name": "storage_tank", "height_cm": 181,
    "base": [x,y],   // lowest point of the tank's front edge, where it touches the floor/plinth
    "top":  [x,y]}},  // highest point of the lid, directly above "base"
   {{"name": "wolf_boiler", "height_cm": 129,
    "base": [x,y],   // middle of the boiler's bottom front edge
    "top":  [x,y]}}   // middle of the boiler's top front edge
 ],
 "step_bottom": [x,y],          // front edge of the raised plinth: where it meets the lower floor
 "step_top": [x,y],             // same edge, top of the step, directly above step_bottom
 "backwall_base": [x,y],        // any visible point where the far wall meets the plinth
 "ceiling_edge": [[x,y],[x,y]], // two points on the line where the ceiling meets the far wall
 "left_wall_on_plinth": [x,y],  // where the left wall meets the plinth surface
 "right_wall_on_plinth": [x,y]  // where the right wall meets the plinth surface
}}"""

DECOUPLED_PROMPT = _HEADER + """
Return exactly this JSON, nothing else:
{{
 "storage_tank_base": [x,y],   // the gray cylindrical hot water tank: lowest point of its front edge, where it touches the floor
 "storage_tank_top":  [x,y],   // same tank: highest point of its lid, directly above storage_tank_base
 "wolf_boiler_base":  [x,y],   // the white Wolf boiler unit: middle of its bottom front edge
 "wolf_boiler_top":   [x,y],   // same boiler: middle of its top front edge, directly above wolf_boiler_base
 "step_bottom": [x,y],          // front edge of the raised plinth: where it meets the lower floor
 "step_top": [x,y],             // same edge, top of the step, directly above step_bottom
 "backwall_base": [x,y],        // any visible point where the far wall meets the plinth
 "ceiling_edge": [[x,y],[x,y]], // two points on the line where the ceiling meets the far wall
 "left_wall_on_plinth": [x,y],  // where the left wall meets the plinth surface
 "right_wall_on_plinth": [x,y]  // where the right wall meets the plinth surface
}}"""

HOMOGRAPHY_PROMPT = _HEADER + """
The back wall of this room is a flat vertical rectangular surface. Find its
4 corners as they appear in the photo (they will look like a distorted
quadrilateral due to camera perspective, not a perfect rectangle -- that is
expected, just report where each corner actually is in the image).

Return exactly this JSON, nothing else:
{{
 "wall_top_left": [x,y],
 "wall_top_right": [x,y],
 "wall_bottom_right": [x,y],
 "wall_bottom_left": [x,y],
 "storage_tank_base": [x,y],   // the gray cylindrical hot water tank: lowest point of its front edge, where it touches the floor
 "storage_tank_top":  [x,y],   // same tank: highest point of its lid, directly above storage_tank_base
 "wolf_boiler_base":  [x,y],   // the white Wolf boiler unit: middle of its bottom front edge
 "wolf_boiler_top":   [x,y]    // same boiler: middle of its top front edge, directly above wolf_boiler_base
}}"""

VARIANTS = {   # name -> (prompt, max_new_tokens)
    "with-sizes": (WITH_SIZES_PROMPT, 600),
    "decoupled": (DECOUPLED_PROMPT, 500),
    "homography": (HOMOGRAPHY_PROMPT, 400),
}
CORNER_KEYS = ["wall_top_left", "wall_top_right", "wall_bottom_right", "wall_bottom_left"]


def is_point(v) -> bool:
    return isinstance(v, list) and len(v) == 2 and all(isinstance(c, (int, float)) for c in v)


def flatten_points(data: dict) -> dict[str, list]:
    """All returned points as {name: [x, y]}, with with-sizes' nested
    "references" list unpacked to the decoupled schema's flat names."""
    points = {}
    for ref in data.get("references") or []:
        for key in ("base", "top"):
            if is_point(ref.get(key)):
                points[f"{ref.get('name', 'ref')}_{key}"] = ref[key]
    for key, val in data.items():
        if is_point(val):
            points[key] = val
        elif key == "ceiling_edge" and isinstance(val, list):
            for i, pt in enumerate(val[:2]):
                if is_point(pt):
                    points[f"ceiling_edge_{i}"] = pt
    return points


def draw_points(img: Image.Image, points: dict[str, list]) -> Image.Image:
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    r = 6
    for label, (x, y) in points.items():
        draw.ellipse([x - r, y - r, x + r, y + r], outline="red", width=2)
        draw.text((x + r + 2, y - r), label, fill="yellow")
    if all(k in points for k in CORNER_KEYS):
        draw.polygon([tuple(points[k]) for k in CORNER_KEYS], outline="cyan")
    return annotated


def flat_scale(points: dict, base_key: str, top_key: str, known_cm: float) -> Optional[float]:
    if base_key in points and top_key in points:
        px = points[base_key][1] - points[top_key][1]
        return known_cm / px if px > 0 else None
    return None


def analyse(points: dict, wall) -> None:
    """Print the reference-scale agreement and whatever estimate the returned
    points support: homography-solved height if the wall corners came back,
    else the naive flat-scale height and width."""
    tank = flat_scale(points, "storage_tank_base", "storage_tank_top", KNOWN_TANK_H_CM)
    boiler = flat_scale(points, "wolf_boiler_base", "wolf_boiler_top", KNOWN_BOILER_H_CM)
    for name, s in (("storage_tank", tank), ("wolf_boiler", boiler)):
        print(f"  {name}: " + (f"{s:.3f} cm/px" if s else "missing or degenerate points"))
    if tank and boiler:
        print(f"  Reference scale disagreement: {abs(tank - boiler) / min(tank, boiler) * 100:.0f}%")

    if all(k in points for k in CORNER_KEYS):   # one reference is enough for the homography height
        refs = {"tank_base": points.get("storage_tank_base"), "tank_top": points.get("storage_tank_top"),
                "boiler_base": points.get("wolf_boiler_base"), "boiler_top": points.get("wolf_boiler_top")}
        result = solve_wall_height([points[k] for k in CORNER_KEYS], refs)
        if result is None:
            print("  Homography solve failed (degenerate reference points)")
            return
        if result["scale_disagreement_pct"] is not None:
            print(f"  Rectified scale disagreement: {result['scale_disagreement_pct']:.0f}%")
        print(f"  Height: {error_report(result['height_cm'], wall.height_cm)}")
        return

    if not (tank and boiler):
        return
    scale = (tank + boiler) / 2   # flat average, no perspective correction
    if "step_bottom" in points and "ceiling_edge_0" in points and "ceiling_edge_1" in points:
        ceiling_y = (points["ceiling_edge_0"][1] + points["ceiling_edge_1"][1]) / 2
        print(f"  Naive room height: {error_report((points['step_bottom'][1] - ceiling_y) * scale, ROOM_HEIGHT_CM)}")
    if "left_wall_on_plinth" in points and "right_wall_on_plinth" in points:
        width_px = abs(points["right_wall_on_plinth"][0] - points["left_wall_on_plinth"][0])
        print(f"  Naive wall width:  {error_report(width_px * scale, wall.width_cm)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()
    prompt_template, max_new_tokens = VARIANTS[args.variant]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, processor = load_qwen(QWEN25_7B)

    results = {}
    for wall in (w for w in WALLS if w.has_tank):
        img = load_pil(IMAGES_DIR / wall.file)
        resp = ask(model, processor, [img], prompt_template.format(W=img.width, H=img.height), max_new_tokens)
        print(f"=== {wall.name} ({wall.file}) — sent as {img.width}x{img.height} ===\n{resp}\n")
        results[wall.name] = {"raw": resp}

        data = parse_json_object(resp)
        if data is None:
            print("  Could not parse JSON from the response.\n")
            continue
        points = flatten_points(data)
        results[wall.name]["points"] = points
        out_path = OUT_DIR / f"{wall.name}_{args.variant}.png"
        draw_points(img, points).save(out_path)
        print(f"  Points drawn -> {out_path}")
        analyse(points, wall)
        print()

    (OUT_DIR / f"{args.variant}.json").write_text(json.dumps(results, indent=2))
    print(f"Outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
