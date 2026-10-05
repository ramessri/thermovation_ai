"""
disk_lightglue_room_dims.py — room L x W x H from a DISK+LightGlue
reconstruction (run_sfm_disk_lightglue.py), judged by the EXACT SAME
compute_room_dims() every other arm in this project uses (SfM, 3DGS, NeRF,
MASt3R, VGGT) — a difference in the resulting numbers reflects the
underlying reconstruction, not a different hand-rolled implementation.

Scale handling: run_sfm_disk_lightglue.py deliberately skips metric scale
(out of scope for that comparison — no marker triangulation attempted).
This script bridges scale the same way 3d_recons_alternatives/
vggt_reconstruct.py does for VGGT's equally-scale-free output: compare
camera-to-camera distances against the EXISTING marker-scaled SIFM
reconstruction for the SAME video, matched by frame filename (both are
plain pycolmap.Reconstruction objects here, so no extrinsics-convention
juggling is needed the way VGGT's raw numpy poses required) — a single
robust scale ratio (median of the tightest 50% cluster, same aggregation
depth_scale.py/vggt_reconstruct.py already use), not a full re-alignment.

Usage:
  python 03_reconstruction/disk_lightglue_room_dims.py \
      --disk-lg-dir 3d_recons_alternatives/disk_lg/Klaus_Rombergg \
      --sfm-dir "C:\\thermovation-output\\sfm 1\\Klaus_Rombergg\\sfm" --sfm-model 0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pycolmap

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "05_geometry"))
from room_dims import load_filtered_points, gravity_rotation, compute_room_dims


def _robust_mode(ratios: np.ndarray) -> tuple[float, float]:
    """Median of the densest 50% cluster + its coefficient of variation —
    same 'tightest-window' aggregation depth_scale.py and
    vggt_reconstruct.py both already use for exactly this kind of ratio."""
    r = np.array(sorted(ratios))
    k = max(int(0.5 * len(r)), 3)
    _, best = min((r[i + k - 1] - r[i], i) for i in range(len(r) - k + 1))
    core = r[best:best + k]
    return float(np.median(core)), float(np.std(core) / np.mean(core) * 100)


def fit_scale_to_sfm(disk_lg_rec: pycolmap.Reconstruction, sfm_rec: pycolmap.Reconstruction,
                     sfm_cm_per_unit: float, max_pairs: int = 500) -> dict | None:
    # run_sfm_disk_lightglue.py hardlinks frames into a space-free staging
    # dir before reconstruction (hloc's pairs.txt parser breaks on this
    # project's space-containing filenames — see that script's
    # sanitize_frames_dir docstring), so DISK+LightGlue image names have
    # underscores where the original SIFT reconstruction's names have
    # spaces. Normalize both sides the same way before matching, or every
    # frame silently fails to match and this returns None.
    name_to_disk_center = {im.name: np.asarray(im.projection_center())
                           for im in disk_lg_rec.images.values()}
    name_to_sfm_center_cm = {im.name.replace(" ", "_"): np.asarray(im.projection_center()) * sfm_cm_per_unit
                             for im in sfm_rec.images.values()}
    common = [n for n in name_to_disk_center if n in name_to_sfm_center_cm]
    if len(common) < 5:
        return None

    rng = np.random.default_rng(0)
    pairs = [(a, b) for ai, a in enumerate(common) for b in common[ai + 1:]]
    if len(pairs) > max_pairs:
        idx = rng.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]

    ratios = []
    for a, b in pairs:
        d_disk = float(np.linalg.norm(name_to_disk_center[a] - name_to_disk_center[b]))
        d_sfm = float(np.linalg.norm(name_to_sfm_center_cm[a] - name_to_sfm_center_cm[b]))
        if d_disk > 1e-9:
            ratios.append(d_sfm / d_disk)
    if len(ratios) < 20:
        return None

    cm_per_unit, cv_pct = _robust_mode(np.array(ratios))
    return dict(cm_per_disk_lg_unit=cm_per_unit, cv_pct=round(cv_pct, 2),
               frames_matched=len(common), pairs_used=len(ratios))


def main():
    parser = argparse.ArgumentParser(description="Room L x W x H from a DISK+LightGlue reconstruction")
    parser.add_argument("--disk-lg-dir", type=Path, required=True,
                        help="Output dir from run_sfm_disk_lightglue.py (contains sparse/)")
    parser.add_argument("--sfm-dir", type=Path, required=True,
                        help="This video's EXISTING SIFT sfm dir (contains sparse/<model> and scale.json) "
                             "— used only to bridge metric scale by camera-trajectory ratio")
    parser.add_argument("--sfm-model", default="0")
    parser.add_argument("--out", type=Path, default=None,
                        help="Default: <disk-lg-dir>/disk_lg_room_dims.json")
    args = parser.parse_args()

    disk_lg_rec = pycolmap.Reconstruction(str(args.disk_lg_dir / "sparse"))
    print(f"DISK+LightGlue model: {disk_lg_rec.num_reg_images()} images, "
          f"{disk_lg_rec.num_points3D()} points")

    scale_info = json.loads((args.sfm_dir / "scale.json").read_text())
    sfm_cm_per_unit = scale_info["cm_per_unit"]
    sfm_rec = pycolmap.Reconstruction(str(args.sfm_dir / "sparse" / args.sfm_model))
    print(f"SIFT reference model: {sfm_rec.num_reg_images()} images, scale "
          f"{sfm_cm_per_unit:.4f} cm/unit")

    fit = fit_scale_to_sfm(disk_lg_rec, sfm_rec, sfm_cm_per_unit)
    if fit is None:
        print("FAILED: fewer than 5 frames in common (or fewer than 20 valid pairwise "
              "distance ratios) between the DISK+LightGlue and SIFT reconstructions — "
              "cannot bridge scale.")
        return
    cm_per_unit = fit["cm_per_disk_lg_unit"]
    print(f"Scale bridge: {fit['frames_matched']} frames matched, "
          f"{fit['pairs_used']} pairs, {cm_per_unit:.4f} cm/unit "
          f"(spread {fit['cv_pct']:.1f}%)")
    if fit["cv_pct"] > 25.0:
        print(f"WARNING: scale-fit spread {fit['cv_pct']:.1f}% is wide — the two "
              f"reconstructions' camera trajectories disagree on shape more than a "
              f"tight calibration should allow. Treat the resulting cm numbers with "
              f"more caution than the other arms' scale bridges.")

    pts = load_filtered_points(disk_lg_rec) * cm_per_unit
    print(f"Filtered points: {len(pts)}")
    R, z_floor = gravity_rotation(disk_lg_rec, pts)
    if z_floor is None:
        print("FAILED: could not localize the floor density peak")
        return

    cam_centers_cm = np.array([im.projection_center() for im in disk_lg_rec.images.values()]) * cm_per_unit
    metrics, geometry = compute_room_dims(pts, R, z_floor, cam_centers_cm)

    bound = ">= " if metrics["height_is_lower_bound"] else ""
    flag_h = "" if metrics["height_reliable"] else "  [LOW CONFIDENCE]"
    flag_f = "" if metrics["footprint_reliable"] else "  [LOW CONFIDENCE]"
    print(f"\nRoom height : {bound}{metrics['height_cm']:.1f} cm{flag_h}")
    print(f"Room length : {metrics['length_cm']:.1f} cm{flag_f}")
    print(f"Room width  : {metrics['width_cm']:.1f} cm{flag_f}")

    out_path = args.out or (args.disk_lg_dir / "disk_lg_room_dims.json")
    result = dict(method="disk_lightglue", scale_source="camera_trajectory_ratio_vs_sfm",
                 cm_per_unit=cm_per_unit, scale_fit=fit, **metrics)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
