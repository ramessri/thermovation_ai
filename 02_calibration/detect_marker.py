"""
detect_marker.py — automatic scale recovery from the custom boiler-room fiducial marker.

Marker physical layout (printed, cm):
  Board outer size : 28.6 x 20.2
  9 black squares  : 3.9 x 3.9, arranged in a regular 3x3 grid
  Center-to-center : 12.35 horizontally, 8.15 vertically

Detection strategy (robust to boiler-room clutter):
  1. Extract dark square-ish blobs at several adaptive-threshold scales.
  2. Search for a 3x3 grid of similar-sized blobs whose center spacing matches
     the known step/side ratios (~3.17 horizontal, ~2.09 vertical).
  3. Validate with a homography fit (low reprojection error) and a darkness
     check (squares must be much darker than the surrounding paper).
  A frame with no valid grid returns "not detected" — no fallback guessing.

Usage as a script:
  python 02_calibration/detect_marker.py image_or_video [--debug] [--output out.jpg] [--every-n 5]

Usage as a module:
  from detect_marker import detect_marker, best_marker_frame, sharpness_score
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

# ── Physical marker constants (cm) ────────────────────────────────────────────
BOARD_W_CM = 28.6
BOARD_H_CM = 20.2
SQUARE_CM  = 3.9
STEP_X_CM = (BOARD_W_CM - SQUARE_CM) / 2      # 12.35 center-to-center, horizontal
STEP_Y_CM = (BOARD_H_CM - SQUARE_CM) / 2      # 8.15  center-to-center, vertical
RATIO_U = STEP_X_CM / SQUARE_CM               # ~3.17  horizontal step / square side
RATIO_V = STEP_Y_CM / SQUARE_CM               # ~2.09  vertical step / square side

# cm coordinates of the 9 square centers, row-major (TL first)
GRID_CM = np.array(
    [[c * STEP_X_CM + SQUARE_CM / 2, r * STEP_Y_CM + SQUARE_CM / 2]
     for r in range(3) for c in range(3)],
    dtype=np.float32,
)

# Downscale very large frames for blob extraction (centroids stay accurate)
MAX_DETECT_DIM = 1920


def sharpness_score(bgr_or_gray: np.ndarray) -> float:
    """Variance of the Laplacian — higher means sharper."""
    gray = bgr_or_gray
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ── stage 1: candidate square blobs ──────────────────────────────────────────

def _square_blobs(gray: np.ndarray) -> list[dict]:
    """Dark, roughly square, well-filled blobs at multiple threshold scales."""
    img_area = gray.shape[0] * gray.shape[1]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    raw = []
    for block in (31, 91, 241):
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block, 10,
        )
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 25 or area > img_area / 9:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if not (0.6 < w / h < 1.67):
                continue
            if area / (w * h) < 0.65:        # squares fill their bounding box
                continue
            raw.append(dict(cx=x + w / 2, cy=y + h / 2, area=area,
                            side=float(np.sqrt(area)), bbox=(x, y, w, h)))
    # dedupe across threshold scales (keep largest of overlapping detections)
    raw.sort(key=lambda c: -c["area"])
    kept: list[dict] = []
    for c in raw:
        if all(np.hypot(c["cx"] - k["cx"], c["cy"] - k["cy"]) > 0.5 * k["side"]
               for k in kept):
            kept.append(c)
    return kept


# ── stage 2: 3x3 grid search ─────────────────────────────────────────────────

def _fit_grid(cands: list[dict]) -> dict | None:
    """Find 9 blobs forming the marker's 3x3 grid. Returns the best hypothesis."""
    if len(cands) < 9:
        return None
    pts = np.array([[c["cx"], c["cy"]] for c in cands], dtype=np.float32)
    sides = np.array([c["side"] for c in cands], dtype=np.float32)
    n = len(cands)
    best: dict | None = None

    for i in range(n):
        s = sides[i]
        rel = pts - pts[i]
        dist = np.linalg.norm(rel, axis=1)
        size_ok = (sides > 0.6 * s) & (sides < 1.67 * s)
        u_idx = np.where(size_ok & (dist > RATIO_U * s * 0.65) & (dist < RATIO_U * s * 1.45))[0]
        v_idx = np.where(size_ok & (dist > RATIO_V * s * 0.65) & (dist < RATIO_V * s * 1.45))[0]

        for j in u_idx:
            u = rel[j]
            for k in v_idx:
                if k == j:
                    continue
                v = rel[k]
                # steps must be roughly perpendicular
                cosang = abs(float(u @ v)) / (dist[j] * dist[k] + 1e-9)
                if cosang > 0.35:
                    continue
                # horizontal step is 1.515x the vertical step (tolerate perspective)
                ratio = dist[j] / dist[k]
                if not (1.05 < ratio < 2.2):
                    continue

                # predict all 9 centers from the anchor + basis vectors
                tol = 0.35 * min(dist[j], dist[k])
                idxs: list[int] = []
                ok = True
                for r in range(3):
                    for c in range(3):
                        pred = pts[i] + c * u + r * v
                        dd = np.linalg.norm(pts - pred, axis=1)
                        m = int(np.argmin(dd))
                        if dd[m] < tol and 0.55 < sides[m] / s < 1.8:
                            idxs.append(m)
                        else:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok or len(set(idxs)) != 9:
                    continue

                grid_pts = pts[idxs]
                H_cm2px, _ = cv2.findHomography(GRID_CM.reshape(-1, 1, 2),
                                                grid_pts.reshape(-1, 1, 2))
                if H_cm2px is None:
                    continue
                proj = cv2.perspectiveTransform(GRID_CM.reshape(-1, 1, 2),
                                                H_cm2px).reshape(-1, 2)
                err = float(np.mean(np.linalg.norm(proj - grid_pts, axis=1)))
                if err > 0.12 * min(dist[j], dist[k]):
                    continue

                # prefer the largest marker appearance (closest / most accurate)
                if best is None or dist[j] > best["step_u_px"]:
                    best = dict(idxs=idxs, grid_pts=grid_pts, H_cm2px=H_cm2px,
                                reproj_err=err, step_u_px=float(dist[j]),
                                step_v_px=float(dist[k]))
    return best


# ── stage 3: photometric validation ──────────────────────────────────────────

def _darkness_ok(gray: np.ndarray, cands: list[dict], idxs: list[int]) -> bool:
    """Squares must be clearly darker than the paper between them."""
    square_mask = np.zeros(gray.shape, dtype=np.uint8)
    square_means = []
    for m in idxs:
        x, y, w, h = cands[m]["bbox"]
        square_means.append(float(gray[y:y + h, x:x + w].mean()))
        cv2.rectangle(square_mask, (x, y), (x + w, y + h), 255, -1)

    xs = [cands[m]["cx"] for m in idxs]
    ys = [cands[m]["cy"] for m in idxs]
    pad = int(cands[idxs[0]]["side"])
    x0 = max(int(min(xs)) - pad, 0)
    x1 = min(int(max(xs)) + pad, gray.shape[1])
    y0 = max(int(min(ys)) - pad, 0)
    y1 = min(int(max(ys)) + pad, gray.shape[0])

    region = gray[y0:y1, x0:x1]
    region_mask = square_mask[y0:y1, x0:x1]
    paper = region[region_mask == 0]
    if paper.size == 0:
        return False
    return float(np.mean(square_means)) < 0.75 * float(paper.mean())


# ── public API ────────────────────────────────────────────────────────────────

def detect_marker(
    image_bgr: np.ndarray,
    debug: bool = False,
) -> tuple[float | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Detect the 3x3 fiducial marker in a BGR image.

    Returns
    -------
    px_per_cm  : float — scale at the marker (None if not detected)
    corners    : (4,2) float32 — centers of the 4 corner squares, TL/TR/BR/BL
                 (grid order; may be 180°-flipped, irrelevant for scale)
    H_img2cm   : (3,3) float64 — maps image pixels to cm on the board plane
    debug_img  : annotated BGR image if debug=True, else None
    """
    gray_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # blob extraction on a downscaled copy for large frames
    scale = 1.0
    gray = gray_full
    if max(gray_full.shape) > MAX_DETECT_DIM:
        scale = MAX_DETECT_DIM / max(gray_full.shape)
        gray = cv2.resize(gray_full, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)

    cands = _square_blobs(gray)
    fit = _fit_grid(cands)
    if fit is None or not _darkness_ok(gray, cands, fit["idxs"]):
        return None, None, None, None

    grid_pts_full = fit["grid_pts"] / scale          # back to full-res pixels

    # scale from the homography's local Jacobian at the board center
    H_cm2px_small = fit["H_cm2px"]
    S = np.diag([1.0 / scale, 1.0 / scale, 1.0])     # small-px -> full-px
    H_cm2px = S @ H_cm2px_small
    center = np.array([[BOARD_W_CM / 2, BOARD_H_CM / 2]], dtype=np.float32)
    probe = np.array([[BOARD_W_CM / 2 + 1, BOARD_H_CM / 2],
                      [BOARD_W_CM / 2, BOARD_H_CM / 2 + 1]], dtype=np.float32)
    p0 = cv2.perspectiveTransform(center.reshape(-1, 1, 2), H_cm2px).reshape(2)
    p12 = cv2.perspectiveTransform(probe.reshape(-1, 1, 2), H_cm2px).reshape(-1, 2)
    px_per_cm = float((np.linalg.norm(p12[0] - p0) + np.linalg.norm(p12[1] - p0)) / 2)

    H_img2cm = np.linalg.inv(H_cm2px)
    corners = grid_pts_full[[0, 2, 8, 6]].astype(np.float32)   # TL TR BR BL

    debug_img = None
    if debug:
        debug_img = image_bgr.copy()
        inv = 1.0 / scale
        for m in fit["idxs"]:
            x, y, w, h = cands[m]["bbox"]
            cv2.rectangle(debug_img, (int(x * inv), int(y * inv)),
                          (int((x + w) * inv), int((y + h) * inv)), (0, 220, 0), 2)
        # grid lines
        gp = grid_pts_full.astype(int)
        for r in range(3):
            for c in range(2):
                cv2.line(debug_img, tuple(gp[r * 3 + c]), tuple(gp[r * 3 + c + 1]),
                         (0, 180, 255), 2)
        for r in range(2):
            for c in range(3):
                cv2.line(debug_img, tuple(gp[r * 3 + c]), tuple(gp[(r + 1) * 3 + c]),
                         (0, 180, 255), 2)
        for pt, label, color in zip(corners, ["TL", "TR", "BR", "BL"],
                                    [(255, 80, 80), (80, 255, 80),
                                     (80, 80, 255), (255, 255, 80)]):
            cv2.circle(debug_img, tuple(pt.astype(int)), 10, color, -1)
            cv2.putText(debug_img, label, (int(pt[0]) + 10, int(pt[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        cv2.putText(debug_img,
                    f"scale: {px_per_cm:.2f} px/cm  reproj_err: {fit['reproj_err'] / scale:.1f}px",
                    (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 240, 240), 2, cv2.LINE_AA)

    return px_per_cm, corners, H_img2cm, debug_img


def _marker_region_sharpness(bgr: np.ndarray, corners: np.ndarray) -> float:
    """Laplacian variance inside the marker bounding box only."""
    x0 = max(int(corners[:, 0].min()), 0)
    x1 = min(int(corners[:, 0].max()), bgr.shape[1])
    y0 = max(int(corners[:, 1].min()), 0)
    y1 = min(int(corners[:, 1].max()), bgr.shape[0])
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0
    return sharpness_score(bgr[y0:y1, x0:x1])


def best_marker_frame(
    video_path: str | Path,
    every_n: int = 5,
    min_sharpness: float = 25.0,
    verbose: bool = True,
) -> tuple[np.ndarray | None, float | None, np.ndarray | None, np.ndarray | None]:
    """
    Scan a video; among frames with a validated marker, return the one where
    the marker appears largest (closest -> most accurate scale), requiring
    minimum sharpness in the marker region.

    Returns (best_frame_bgr, px_per_cm, corners, H_img2cm) — all None if the
    marker never appears.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    best = dict(frame=None, px_per_cm=-1.0, corners=None, H=None, sharp=0.0)
    checked = detected = 0
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % every_n == 0:
            checked += 1
            px_per_cm, corners, H, _ = detect_marker(bgr)
            if px_per_cm is not None:
                detected += 1
                sharp = _marker_region_sharpness(bgr, corners)
                if sharp >= min_sharpness and px_per_cm > best["px_per_cm"]:
                    best.update(frame=bgr, px_per_cm=px_per_cm,
                                corners=corners, H=H, sharp=sharp)
        idx += 1
    cap.release()

    if verbose:
        print(f"  frames checked: {checked}   marker validated in: {detected}")
    if best["frame"] is None:
        return None, None, None, None
    return best["frame"], best["px_per_cm"], best["corners"], best["H"]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Detect boiler-room scale marker")
    parser.add_argument("input", type=Path, help="Image or video file")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--every-n", type=int, default=5,
                        help="(video) sample every Nth frame (default 5)")
    args = parser.parse_args()

    is_video = args.input.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".mts", ".m4v"}

    if is_video:
        print(f"Scanning video: {args.input}")
        frame, px_per_cm, corners, H = best_marker_frame(args.input, every_n=args.every_n)
        if frame is None:
            print("RESULT: marker NOT detected in any frame.")
            return
        _, _, _, debug_img = detect_marker(frame, debug=True)
        print(f"px_per_cm : {px_per_cm:.3f}  ({10 / px_per_cm:.2f} mm/px)")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.output), debug_img if args.debug else frame)
            print(f"Saved to {args.output}")
    else:
        bgr = cv2.imread(str(args.input))
        if bgr is None:
            raise FileNotFoundError(args.input)
        px_per_cm, corners, H, debug_img = detect_marker(bgr, debug=True)
        if px_per_cm is None:
            print("RESULT: marker NOT detected.")
            return
        print(f"px_per_cm : {px_per_cm:.3f}  ({10 / px_per_cm:.2f} mm/px)")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.output), debug_img if args.debug else bgr)
            print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
