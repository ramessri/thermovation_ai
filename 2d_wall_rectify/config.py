"""
Shared configuration for the 2D wall-rectification experiment: dataset
paths, the test room's ground truth, per-wall specs, and the manual
overrides the detectors needed on this photo set.

Photos are read from WALL_PHOTOS_DIR if set, else from
"<DATASET_DIR>/Quirin room images" (DATASET_DIR from the repo's .env, same
as run.py). Outputs go to output/wall_rectify/ (gitignored).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:   # the Molmo venv (see README) has no python-dotenv; plain KEY=VALUE lines suffice
    if (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.strip().startswith("#"):
                os.environ.setdefault(key.strip(), value.split("#")[0].strip())

# Console output uses "—" and "ü"; Windows pipes default to cp1252.
sys.stdout.reconfigure(encoding="utf-8")


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


IMAGES_DIR = _resolve(os.environ.get("WALL_PHOTOS_DIR")
                      or Path(os.environ.get("DATASET_DIR", "dataset")) / "Quirin room images")
OUTPUT_DIR = ROOT / "output" / "wall_rectify"
CORNERS_FILE = HERE / "corners.json"

# ── Ground truth ────────────────────────────────────────────────────────────
# Room: 392 x 249 x 229 cm (L x W x H). Photos alternate width/length walls
# going round the room (confirmed by the user, not inferred).
ROOM_HEIGHT_CM = 229.0

# Fixtures of known real size — the independent scale check. The tank is the
# buffer/storage tank next to the Wolf boiler on walls 3/4.
KNOWN_BOILER_W_CM, KNOWN_BOILER_H_CM = 56.0, 129.0
KNOWN_TANK_H_CM = 181.0

ARUCO_MARKER_SIZE_CM = 15.0   # same default as 02_calibration/detect_marker_aruco.py


@dataclass
class WallSpec:
    name: str
    file: str
    width_cm: float
    height_cm: float
    adjacent_to: Optional[str] = None   # the one confirmed-adjacent wall, if any
    shared_edge: Optional[str] = None   # "left"/"right" edge of THIS wall at the shared corner
    has_tank: bool = False              # reference objects in frame (used by the room-dimension tests)
    has_boiler: bool = False
    corners_px: Optional[list[list[float]]] = None   # TL, TR, BR, BL in original-image px, from corners.json


# Walls 3 & 4 are the one confirmed-adjacent pair: they share the corner
# where the tank and the Wolf boiler sit.
WALLS = [
    WallSpec("wall1", "Image.jfif", 249, ROOM_HEIGHT_CM),
    WallSpec("wall2", "Image (1).jfif", 392, ROOM_HEIGHT_CM),
    WallSpec("wall3", "Image (2).jfif", 249, ROOM_HEIGHT_CM,
             adjacent_to="wall4", shared_edge="right", has_tank=True, has_boiler=True),
    WallSpec("wall4", "Image (3).jfif", 392, ROOM_HEIGHT_CM,
             adjacent_to="wall3", shared_edge="right", has_tank=True),
]
WALLS_BY_NAME = {w.name: w for w in WALLS}

# Wall2 shows neither a marker nor the boiler, so its scale can't be cross-checked.
LOW_CONFIDENCE_WALLS = {"wall2": "no marker and no boiler in frame to cross-check the scale"}

# ── Manual overrides ────────────────────────────────────────────────────────
# Rücklauf on wall3: the blue return dials of the 2-circuit manifold (black
# box with 2 blue + 2 red gauges), midpoint of the two blue dials, in
# original-image px. The HSV detector can't isolate these small, dim dials
# from JPEG/reflection noise even with loosened thresholds (README §4.2).
MANUAL_RUCKLAUF_PX: dict[str, tuple[float, float]] = {
    "wall3": (1900.0, 1470.0),
}

# Obstacles GroundingDINO missed. Without these keep-outs the search put the
# unit on the manifold's valves, then the boiler's front panel, then the
# tank — each the free point nearest the Rücklauf once nearer obstacles were
# excluded. Boxes are [x0, y0, x1, y1] in original-image px.
MANUAL_OBSTACLE_BOXES: dict[str, list[dict]] = {
    "wall3": [
        {"label": "manifold (manual)", "box": [1650, 1300, 2090, 1650]},
        {"label": "boiler (manual)", "box": [2380, 900, 3050, 2480]},
        {"label": "hot water tank (manual)", "box": [680, 620, 1240, 2420]},
    ],
    "wall4": [
        {"label": "hot water tank (manual)", "box": [2650, 650, 3110, 2500]},
    ],
}


def load_walls() -> list[WallSpec]:
    """WALLS with corners_px filled in from corners.json (written by corner_picker.py)."""
    if not CORNERS_FILE.exists():
        raise FileNotFoundError(f"{CORNERS_FILE} not found — run corner_picker.py and click each wall's corners first")
    corners = json.loads(CORNERS_FILE.read_text())
    missing = [w.name for w in WALLS if w.name not in corners]
    if missing:
        raise ValueError(f"{CORNERS_FILE} has no corners for {missing} — re-run corner_picker.py")
    for wall in WALLS:
        wall.corners_px = corners[wall.name]
    return WALLS
