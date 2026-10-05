"""
detect_marker_aruco.py — ArUco marker detection, an alternative scale
reference to the custom 3x3 board in detect_marker.py.

Unlike the custom board (one fixed marker, moved between walls during
filming), ArUco markers are OpenCV's built-in fiducial system: any number
of small individually-numbered squares (DICT_4X4_100 by default) that can
be printed and scattered around the scene. Detection uses cv2.aruco's own
detector rather than the custom board's blob/grid-fitting pipeline — no
photometric/darkness validation needed, the dictionary decode itself is
the validation (a false positive would need to decode a valid marker ID,
which random clutter essentially never does).

Marker physical size is a single side length (ARUCO_MARKER_SIZE_CM, default
15cm — matches the sibling photogram project's ARUCO_MARKER_SIZE_M=0.15
default) rather than a fixed multi-square grid.

Usage as a module:
    from detect_marker_aruco import detect_aruco_markers, ARUCO_CORNERS_CM

Usage as a script:
    python 02_calibration/detect_marker_aruco.py image.jpg [--debug] [--output out.jpg]
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

ARUCO_MARKER_SIZE_CM = float(os.environ.get("ARUCO_MARKER_SIZE_CM", "15.0"))
ARUCO_DICT_NAME = os.environ.get("ARUCO_DICT", "DICT_4X4_100")

_DICT_MAP = {
    "DICT_4X4_50":  cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
}

# 4 corners of a marker in its own local cm frame, z=0, in cv2.aruco's own
# corner ordering (TL, TR, BR, BL going clockwise from top-left).
h = ARUCO_MARKER_SIZE_CM / 2.0
ARUCO_CORNERS_CM = np.array([[-h, h], [h, h], [h, -h], [-h, -h]], dtype=np.float32)


def _get_detector() -> "cv2.aruco.ArucoDetector":
    dict_id = _DICT_MAP.get(ARUCO_DICT_NAME, cv2.aruco.DICT_4X4_100)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(aruco_dict, params)


def detect_aruco_markers(image_bgr: np.ndarray) -> dict[int, np.ndarray]:
    """
    Detect all ArUco markers in a BGR image.

    Returns {marker_id: corners} where corners is (4,2) float32 in image
    pixels, TL/TR/BR/BL order (cv2.aruco's native convention).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    detector = _get_detector()
    corners_list, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return {}
    return {int(mid): corners[0].astype(np.float32)
            for corners, mid in zip(corners_list, ids.flatten())}


def main():
    parser = argparse.ArgumentParser(description="Detect ArUco markers in an image")
    parser.add_argument("input", type=Path)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    bgr = cv2.imread(str(args.input))
    if bgr is None:
        raise FileNotFoundError(args.input)

    dets = detect_aruco_markers(bgr)
    if not dets:
        print("RESULT: no ArUco markers detected.")
        return
    print(f"RESULT: {len(dets)} marker(s) — IDs {sorted(dets)} "
         f"(dict={ARUCO_DICT_NAME}, size={ARUCO_MARKER_SIZE_CM}cm)")

    if args.debug or args.output:
        vis = bgr.copy()
        for mid, corners in dets.items():
            pts = corners.astype(int)
            cv2.polylines(vis, [pts], True, (0, 220, 0), 2)
            cx, cy = corners.mean(axis=0).astype(int)
            cv2.putText(vis, str(mid), (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 180, 255), 2, cv2.LINE_AA)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.output), vis)
            print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
