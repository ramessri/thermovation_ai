"""
calibrate_camera.py — Zhang-style camera calibration using the fiducial marker
as a planar calibration target (the same principle as chessboard calibration).

The given marker's 3x3 grid has known metric geometry (detect_marker.GRID_CM), and the video
naturally observes it from many different angles as the camera walks around
— this is needed for the 2D<->3D correspondences by cv2.calibrateCamera.

Camera model is COLMAP's "OPENCV" model (fx, fy, cx, cy, k1, k2, p1, p2)
— chosen because it uses the identical distortion convention as
cv2.calibrateCamera, so the calibration transplants directly with no
reparameterization.

Usage:
  python 02_calibration/calibrate_camera.py dataset/frames/<video_(sfm)> --output output/calib/video.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from detect_marker import detect_marker, GRID_CM

MIN_VIEWS = 15
MAX_VIEWS = 80

# Plausibility gates — reject a calibration when the marker is only ever seen
# from a narrow range of tilts/positions (ill-conditioned for Zhang's method:
# the optimizer can trade off focal length against high-order distortion to
# fit the limited training views, producing a low-residual but nonsensical
# lens model that generalizes badly to the rest of the video).
MAX_ABS_K1 = 0.3
MAX_ABS_K2 = 0.6
MAX_PRINCIPAL_OFFSET_FRAC = 0.20   # cx/cy must be within 20% of true image center
MIN_TILT_SPREAD_DEG = 12.0         # marker plane normal must vary this much across views
MIN_DFOV_DEG = 65.0                # diagonal FOV floor — rejects focal lengths implying
MAX_DFOV_DEG = 130.0               # 2×+ telephoto or fisheye (implausible for site-survey captures)


def collect_views(frames_dir: Path):
    obj_pts, img_pts, names = [], [], []
    image_size = None
    for f in sorted(frames_dir.glob("*.jpg")):
        bgr = cv2.imread(str(f))
        if bgr is None:
            continue
        if image_size is None:
            image_size = (bgr.shape[1], bgr.shape[0])
        px_per_cm, _, _, grid_pts_full, _ = detect_marker(bgr)
        if px_per_cm is None:
            continue
        obj = np.hstack([GRID_CM, np.zeros((9, 1))]).astype(np.float32)
        obj_pts.append(obj)
        img_pts.append(grid_pts_full.astype(np.float32))
        names.append(f.name)
    return obj_pts, img_pts, image_size, names


def stratified_subsample(obj_pts, img_pts, names, max_views=MAX_VIEWS):
    """Keep temporal diversity (near/far, different tilts along the walk)
    instead of the first N frames, which tend to be similar to each other."""
    n = len(obj_pts)
    if n <= max_views:
        return obj_pts, img_pts, names
    idx = np.linspace(0, n - 1, max_views).round().astype(int)
    idx = sorted(set(idx.tolist()))
    return [obj_pts[i] for i in idx], [img_pts[i] for i in idx], [names[i] for i in idx]


def tilt_spread_deg(rvecs) -> float:
    """Angular spread of the marker-plane normal across calibration views —
    a proxy for view diversity. Low spread means all views are near
    fronto-parallel, which poorly constrains focal length + distortion."""
    normals = []
    for rvec in rvecs:
        Rm, _ = cv2.Rodrigues(rvec)
        normals.append(Rm[:, 2])   # marker's local Z axis in camera frame
    normals = np.array(normals)
    mean_n = normals.mean(axis=0)
    mean_n /= np.linalg.norm(mean_n)
    cosang = np.clip(normals @ mean_n, -1, 1)
    angles = np.degrees(np.arccos(cosang))
    return float(angles.max())


def validate_calibration(result: dict, rvecs) -> str | None:
    """Return a rejection reason string, or None if the calibration passes
    plausibility gates. A low reprojection error alone does not guarantee a
    physically valid lens model — see module docstring."""
    w, h = result["image_width"], result["image_height"]
    fx, fy, cx, cy, k1, k2, p1, p2 = result["camera_params"]
    if abs(k1) > MAX_ABS_K1 or abs(k2) > MAX_ABS_K2:
        return f"implausible distortion (k1={k1:.3f}, k2={k2:.3f})"
    if abs(cx - w / 2) > MAX_PRINCIPAL_OFFSET_FRAC * w:
        return f"principal point cx={cx:.0f} far from image center ({w/2:.0f})"
    if abs(cy - h / 2) > MAX_PRINCIPAL_OFFSET_FRAC * h:
        return f"principal point cy={cy:.0f} far from image center ({h/2:.0f})"
    spread = tilt_spread_deg(rvecs)
    if spread < MIN_TILT_SPREAD_DEG:
        return f"insufficient view-angle diversity (tilt spread {spread:.1f}deg < {MIN_TILT_SPREAD_DEG}deg)"
    diag_px = np.hypot(w, h)
    dfov_deg = 2 * np.degrees(np.arctan(diag_px / (2 * fx)))
    if dfov_deg < MIN_DFOV_DEG or dfov_deg > MAX_DFOV_DEG:
        return (f"implausible diagonal FOV {dfov_deg:.1f}° (focal={fx:.0f}px, diag={diag_px:.0f}px); "
                f"expected {MIN_DFOV_DEG:.0f}–{MAX_DFOV_DEG:.0f}° for a site-survey phone camera — "
                "optimizer likely traded focal length against distortion on limited view diversity")
    return None


def calibrate(frames_dir: Path):
    obj_pts, img_pts, image_size, names = collect_views(frames_dir)
    print(f"  marker detected in {len(obj_pts)} frames")
    if len(obj_pts) < MIN_VIEWS:
        return None
    obj_pts, img_pts, names = stratified_subsample(obj_pts, img_pts, names)
    print(f"  calibrating from {len(obj_pts)} views (stratified subsample)")

    flags = cv2.CALIB_FIX_ASPECT_RATIO
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, image_size, None, None, flags=flags
    )
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    k1, k2, p1, p2 = dist.ravel()[:4]

    # per-view reprojection error spread — flags a poorly-conditioned fit
    # (e.g. all views too fronto-parallel to constrain focal length)
    per_view_err = []
    for i in range(len(obj_pts)):
        proj, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], K, dist)
        e = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img_pts[i]) ** 2, axis=1))))
        per_view_err.append(e)

    result = dict(
        image_width=image_size[0], image_height=image_size[1],
        camera_model="OPENCV",
        camera_params=[float(fx), float(fy), float(cx), float(cy),
                       float(k1), float(k2), float(p1), float(p2)],
        rms_reprojection_error_px=round(float(rms), 3),
        n_views_used=len(obj_pts),
        n_views_available=len(names),
        per_view_error_min=round(min(per_view_err), 3),
        per_view_error_max=round(max(per_view_err), 3),
        tilt_spread_deg=round(tilt_spread_deg(rvecs), 1),
    )
    rejection = validate_calibration(result, rvecs)
    if rejection:
        print(f"  REJECTED: {rejection}")
        print(f"  (had it passed: focal={fx:.0f}px RMS={rms:.3f}px — "
              "low residual error does not guarantee a valid lens model "
              "when view diversity is this limited)")
        return None
    return result


def main():
    parser = argparse.ArgumentParser(description="Calibrate camera intrinsics from marker sightings")
    parser.add_argument("frames_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    print(f"Calibrating from: {args.frames_dir}")
    result = calibrate(args.frames_dir)
    if result is None:
        print("FAILED: no usable calibration (too few marker sightings, or "
              "rejected by plausibility gate — see reason above)")
        return

    print(f"  focal (fx,fy): {result['camera_params'][0]:.1f}, {result['camera_params'][1]:.1f} px")
    print(f"  principal pt : {result['camera_params'][2]:.1f}, {result['camera_params'][3]:.1f}")
    print(f"  distortion   : k1={result['camera_params'][4]:.4f} k2={result['camera_params'][5]:.4f} "
          f"p1={result['camera_params'][6]:.4f} p2={result['camera_params'][7]:.4f}")
    print(f"  RMS reprojection error: {result['rms_reprojection_error_px']:.3f} px "
          f"(per-view range {result['per_view_error_min']:.2f}-{result['per_view_error_max']:.2f})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
