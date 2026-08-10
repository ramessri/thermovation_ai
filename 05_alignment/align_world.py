"""
align_world.py — align a scaled COLMAP sparse model to a real-world "room"
coordinate frame anchored on the fiducial marker.

  read sparse COLMAP model
          |
  read scale.json
          |
  take best marker segment
          |
  marker_points_model <-> GRID_CM
          |
  estimate s, R, t   (3D similarity / Umeyama)
          |
  apply Sim3 transformation
          |
  save output/aligned/sparse/

Solves P_room = s * R @ P_colmap + t for the scale s, rotation R and
translation t that best map the marker's 9 triangulated grid points
(COLMAP model units) onto their known cm layout (GRID_CM, marker plane =
z=0). The same Sim3 is then applied to every point and camera pose in the
reconstruction, so the aligned model is in centimeters with the marker
plane as the world XY plane.

"Best" segment = the accepted marker_triangulation segment with the lowest
internal scale spread (ties broken by more views) — the placement whose own
9-point triangulation was most self-consistent.

The sparse model to align is not user-selected: it's read from scale.json's
own "model_dir" field, since the marker's 3D points only make sense in the
exact model they were triangulated in (a mismatched model would silently
get a wrong-but-plausible-looking transform applied).

Usage:
  python 05_alignment/align_world.py output/pipeline/<video>/sfm
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_calibration"))
from detect_marker import GRID_CM


def pick_best_segment(scale_info: dict) -> dict | None:
    segments = scale_info.get("segments", [])
    if not segments:
        return None
    return min(segments, key=lambda s: (s["spread_pct"], -s["n_views"]))


def fit_planar_sim3(model_pts: np.ndarray, grid_xy: np.ndarray) -> pycolmap.Sim3d:
    """
    3D similarity fit specialized for the marker case: model_pts is a set of
    (near-)coplanar 3D points and grid_xy their known 2D layout on that plane
    (target z=0). The general Umeyama/Kabsch SVD estimator is degenerate for
    coplanar correspondences (COLMAP's estimate_sim3d returns None on this
    input) because the third principal axis carries ~no signal — so instead
    the in-plane 2D similarity (well-conditioned) is fit directly, and the
    out-of-plane axis is fixed only by the chirality (no mirrored world).
    """
    mean_model = model_pts.mean(axis=0)
    _, _, vt3 = np.linalg.svd(model_pts - mean_model)      # vt3[2] = plane normal
    proj = (model_pts - mean_model) @ vt3[:2].T             # in-plane 2D coords

    n = len(proj)
    mu_s, mu_t = proj.mean(axis=0), grid_xy.mean(axis=0)
    sc, tc = proj - mu_s, grid_xy - mu_t
    sigma = (tc.T @ sc) / n
    U, D, Vt = np.linalg.svd(sigma)
    R2 = U @ Vt                                              # 2D fit, reflection allowed
    scale = D.sum() / (sc ** 2).sum(axis=1).mean()
    t2 = mu_t - scale * R2 @ mu_s

    z_sign = np.linalg.det(R2) * np.linalg.det(vt3)          # keep the 3D map a proper rotation
    rot_block = np.eye(3)
    rot_block[:2, :2] = R2
    rot_block[2, 2] = z_sign
    R_full = rot_block @ vt3
    translation = np.append(t2, 0.0) - scale * R_full @ mean_model

    return pycolmap.Sim3d(float(scale), pycolmap.Rotation3d(R_full), translation)


def main():
    parser = argparse.ArgumentParser(
        description="Align a scaled COLMAP model to the marker's room coordinate frame")
    parser.add_argument("sfm_dir", type=Path,
                        help="SfM output dir containing sparse/<model> and scale.json")
    args = parser.parse_args()

    scale_info = json.loads((args.sfm_dir / "scale.json").read_text())
    segment = pick_best_segment(scale_info)
    if segment is None:
        print(f"FAILED: no marker segments in scale.json (method={scale_info.get('method')}) "
              "— alignment needs the marker's own triangulated points.")
        return
    print(f"Best segment: {segment['n_views']} views, spread={segment['spread_pct']:.2f}%, "
          f"[{segment['frames'][0]} .. {segment['frames'][-1]}]")

    src = np.array([segment["marker_points_model"][str(k)] for k in range(9)],
                    dtype=np.float64)                                  # COLMAP model units
    grid_xy = GRID_CM.astype(np.float64)                                # cm, marker plane z=0
    tgt = np.concatenate([grid_xy, np.zeros((9, 1))], axis=1)

    sim3 = fit_planar_sim3(src, grid_xy)
    residual = np.linalg.norm(sim3 * src - tgt, axis=1)
    print(f"Estimated scale: {sim3.scale:.4f} cm/unit "
          f"(scale.json: {segment['cm_per_unit']:.4f} cm/unit)")
    print(f"Marker-point alignment residual: mean={residual.mean():.3f}cm "
          f"max={residual.max():.3f}cm")

    model_name = Path(scale_info["model_dir"]).name
    model_dir = args.sfm_dir / "sparse" / model_name
    rec = pycolmap.Reconstruction(str(model_dir))
    print(f"Model: {rec.num_reg_images()} images, {rec.num_points3D()} points")
    rec.transform(sim3)

    out_dir = args.sfm_dir / "aligned" / "sparse"
    out_dir.mkdir(parents=True, exist_ok=True)
    rec.write(str(out_dir))
    print(f"Saved aligned model to {out_dir}")


if __name__ == "__main__":
    main()
