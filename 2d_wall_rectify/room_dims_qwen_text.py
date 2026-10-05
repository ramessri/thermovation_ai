"""
Can a local Qwen VLM estimate the room's dimensions by itself, answering in
text, given the boiler (129x56cm) and tank (181cm) as visual scale
references? Five prompt variants, run in the order they were tried:

  2b-json        Qwen2-VL-2B, per wall, direct answer as JSON.
                 -> ~240cm height on every wall whether or not the references
                    were visible (a stock "typical room" prior); width 27-54% off.
  7b-cot         Qwen2.5-VL-7B (4-bit), per wall, forced step-by-step
                 reasoning (reference fraction of frame -> scale -> answer).
  7b-plain       7B, all 4 photos in one message, one-line question
                 ("find the dimensions of the room") — what worked with Claude Opus.
  7b-structured  7B, all 4 photos, the full structured prompt used with Opus
                 (orientation -> scale -> height -> width -> uncertainty).
  7b-boiler-only 7B, one photo per call at a higher resolution cap, only the
                 boiler's size given.

All variants failed at the arithmetic: unit slips (metres for cm), regression
to generic-room priors, and reasoning text that contradicts the final number.
That led to the pointing-only approach in room_dims_qwen_pointing.py.

Usage: python 2d_wall_rectify/room_dims_qwen_text.py --variant 7b-cot
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from typing import Optional

from config import IMAGES_DIR, OUTPUT_DIR, ROOM_HEIGHT_CM, WALLS
from room_dims_solver import error_report
from vlm import MAX_PIXELS, QWEN2_2B, QWEN25_7B, ask, load_pil, load_qwen

OUT_DIR = OUTPUT_DIR / "room_dims_qwen_text"

JSON_PROMPT = (
    "This is a photo of a boiler room. Two objects in it have known real-world size: "
    "a white Wolf gas boiler unit that is 129 cm tall and 56 cm wide, and a gray "
    "cylindrical hot water storage tank that is 181 cm tall. "
    "Using these two objects as a visual scale reference, estimate: "
    "(1) the room's floor-to-ceiling height, and "
    "(2) the width of the wall shown, measured corner-to-corner. "
    "IMPORTANT: give both answers as a number of CENTIMETERS, not meters — "
    "for example a typical room height would be written as 240, not 2.4. "
    'Respond ONLY with JSON: {"room_height_cm": <number, centimeters>, '
    '"wall_width_cm": <number, centimeters>, "reasoning": "<one short sentence>"}'
)

COT_PROMPT = (
    "This is a photo of a boiler room. Two objects in it have known real-world size: "
    "a white Wolf gas boiler unit that is 129 cm tall and 56 cm wide, and a gray "
    "cylindrical hot water storage tank that is 181 cm tall.\n\n"
    "Work through this step by step, in order, before giving your final answer:\n"
    "1. Identify whether the boiler and/or tank are actually visible in this image. "
    "If neither is visible, say so explicitly.\n"
    "2. For whichever reference object IS visible, estimate what fraction of the "
    "image's vertical height its known real-world height occupies (e.g. \"the boiler "
    "spans about 30% of the frame height\").\n"
    "3. Using that fraction as your pixel-to-cm scale, estimate what fraction of the "
    "frame the floor-to-ceiling extent occupies, and convert that to centimeters.\n"
    "4. Using the same scale, estimate what fraction of the frame's width the visible "
    "wall spans corner-to-corner, and convert that to centimeters.\n\n"
    "IMPORTANT: give both final answers as a number of CENTIMETERS, not meters -- "
    "for example a typical room height would be written as 240, not 2.4.\n\n"
    "After your step-by-step reasoning, end your response with ONLY this JSON on the "
    "final line: {\"room_height_cm\": <number, centimeters>, "
    "\"wall_width_cm\": <number, centimeters>, \"reasoning\": \"<one short sentence>\"}"
)

PLAIN_PROMPT = (
    "The Wolf gas boiler in these photos is 129 cm tall and 56 cm wide. "
    "Find the dimensions of the room."
)

BOILER_ONLY_PROMPT = (
    "The Wolf gas boiler visible in these photos is 129 cm tall and 56 cm wide. "
    "Find the dimensions of the room."
)

STRUCTURED_PROMPT = """I'm attaching 4 photos of a room taken from different positions.
Estimate the room's dimensions (length, width, ceiling height, floor area).

Known reference sizes:
- Wolf gas boiler: 129 cm tall, 56 cm wide
- Gray cylindrical hot water storage/buffer tank: 181 cm tall

Work through it in this order and show your reasoning:

1. Orientation: For each photo, say which wall it faces and which walls
   appear left and right. Use objects visible in more than one photo
   (doors, pipes, furniture) to link the views into a single floor plan.
   State the layout before measuring anything.

2. Scale: For each reference object, estimate its span in pixels and
   calculate pixels per cm at that object's distance from the camera.
   Note that walls further away than the reference will have a smaller
   scale, and adjust for that.

3. Height: Use the tallest upright reference close to a wall. Add the
   gap to the ceiling and any plinth or raised floor.

4. Width and length: Add up the objects and gaps along each wall. Where
   possible, also measure each wall a second way (a direct corner-to-corner
   measurement, or a different reference object) and compare the results.

5. Secondary references: If you infer sizes for objects I didn't give
   (doors, tiles, hatches, equipment), list each assumed size and say
   why you assumed it.

6. Uncertainty: Give each dimension as a range. Say which figure is most
   and least reliable, and why (lens distortion, perspective, assumed sizes).

Finish with a summary table and a short list of which single extra
measurement would improve accuracy the most."""


@dataclass
class Variant:
    model_id: str
    prompt: str
    max_new_tokens: int
    all_photos_at_once: bool = False   # one message with all 4 photos, else one call per wall
    scored_json: bool = False          # parse a {"room_height_cm", "wall_width_cm"} answer and score it
    max_dim: Optional[int] = 1024      # pre-resize; None = send native resolution
    max_pixels: int = MAX_PIXELS


VARIANTS = {
    "2b-json": Variant(QWEN2_2B, JSON_PROMPT, 150, scored_json=True),
    "7b-cot": Variant(QWEN25_7B, COT_PROMPT, 500, scored_json=True),
    "7b-plain": Variant(QWEN25_7B, PLAIN_PROMPT, 400, all_photos_at_once=True),
    "7b-structured": Variant(QWEN25_7B, STRUCTURED_PROMPT, 1500, all_photos_at_once=True),
    # All 4 native-res photos in one message (~10k image tokens) OOM'd 10GB, so one per call.
    "7b-boiler-only": Variant(QWEN25_7B, BOILER_ONLY_PROMPT, 400, max_dim=None, max_pixels=2000 * 28 * 28),
}

_ANSWER_RE = re.compile(r'\{[^{}]*"room_height_cm"[^{}]*\}', re.DOTALL)


def parse_answer(raw_text: str) -> Optional[dict]:
    """The last JSON answer block in the response (CoT responses end with it)."""
    for candidate in reversed(_ANSWER_RE.findall(raw_text)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def check(estimate, known_cm: float) -> str:
    try:
        est = float(estimate)
    except (TypeError, ValueError):
        return f"n/a (not a number: {estimate!r})"
    if est < 20:   # answered in metres despite the instruction
        return error_report(est * 100, known_cm) + " (answered in metres, converted)"
    return error_report(est, known_cm)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()
    v = VARIANTS[args.variant]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, processor = load_qwen(v.model_id, max_pixels=v.max_pixels)

    results: dict[str, dict] = {}
    if v.all_photos_at_once:
        images = [load_pil(IMAGES_DIR / w.file, v.max_dim) for w in WALLS]
        resp = ask(model, processor, images, v.prompt, v.max_new_tokens)
        print(f"=== All {len(images)} photos in one message ===\n{resp}\n")
        results["all_walls"] = {"raw": resp}
    else:
        for wall in WALLS:
            img = load_pil(IMAGES_DIR / wall.file, v.max_dim)
            resp = ask(model, processor, [img], v.prompt, v.max_new_tokens)
            print(f"=== {wall.name} ({wall.file}) — tank/boiler visible: {wall.has_tank} ===")
            print(f"{resp}\n")
            results[wall.name] = {"raw": resp}
            if not v.scored_json:
                continue
            answer = parse_answer(resp)
            if answer is None:
                print("  Could not parse a JSON answer.\n")
                continue
            results[wall.name].update(
                parsed=answer,
                height_check=check(answer.get("room_height_cm"), ROOM_HEIGHT_CM),
                width_check=check(answer.get("wall_width_cm"), wall.width_cm),
            )
            print(f"  Room height: {results[wall.name]['height_check']}")
            print(f"  Wall width:  {results[wall.name]['width_check']}\n")

    print("Ground truth: room 392 x 249 x 229 cm; walls 1/3 are 249cm wide, walls 2/4 are 392cm wide")
    out_path = OUT_DIR / f"{args.variant}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
