"""
The inverse of wall_rectify.py: given a wall's 4 corner points but NOT its
size, recover its height from reference objects of known height (tank,
boiler) — the arithmetic half of the VLM room-dimension tests, done in
Python because every VLM tried got the arithmetic wrong (README §5.2).

Rectify the corners to a square canvas: the homography removes perspective,
so a vertical reference's rectified length gives a cm/px scale valid for the
whole wall, and height = canvas height x scale.

Only HEIGHT is recoverable this way. The wall's true aspect ratio is
unknown, and choosing a different one just stretches the canvas vertically
— every vertical length scales by the same factor, so heights stay
consistent but any width derived from vertical references is arbitrary.
(The original version grid-searched the aspect ratio for "boiler/tank
agreement"; that agreement is identical at every ratio, so its widths were
meaningless.) Width needs a horizontal reference, e.g. the boiler's 56cm.

With both references, their scale disagreement is a self-check on the
pointing: it caught the cm-values-echoed-as-pixels contamination.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from config import KNOWN_BOILER_H_CM, KNOWN_TANK_H_CM
from wall_rectify import warp_point

CANVAS = 1000.0


def _ref_scale(H: np.ndarray, base, top, known_h_cm: float) -> Optional[float]:
    if base is None or top is None:
        return None
    px = abs(warp_point(H, base)[1] - warp_point(H, top)[1])
    return known_h_cm / px if px > 1e-6 else None


def solve_wall_height(corners_px, refs: dict) -> Optional[dict]:
    """corners_px: TL, TR, BR, BL. refs: tank_base/top and boiler_base/top
    (either object may be missing). None if no usable reference."""
    dst = np.array([[0, 0], [CANVAS, 0], [CANVAS, CANVAS], [0, CANVAS]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(np.array(corners_px, dtype=np.float32), dst)
    tank = _ref_scale(H, refs.get("tank_base"), refs.get("tank_top"), KNOWN_TANK_H_CM)
    boiler = _ref_scale(H, refs.get("boiler_base"), refs.get("boiler_top"), KNOWN_BOILER_H_CM)
    scales = [s for s in (tank, boiler) if s is not None]
    if not scales:
        return None
    return {
        "tank_cm_per_px": tank,
        "boiler_cm_per_px": boiler,
        "scale_disagreement_pct": abs(tank - boiler) / min(tank, boiler) * 100 if len(scales) == 2 else None,
        "height_cm": CANVAS * float(np.mean(scales)),
    }


def error_report(estimate_cm: float, known_cm: float) -> str:
    err = abs(estimate_cm - known_cm) / known_cm * 100
    tag = "PASS" if err < 10 else "WARN" if err < 25 else "FAIL"
    return f"{estimate_cm:.0f}cm vs known {known_cm:.0f}cm — {err:.0f}% error — {tag}"
