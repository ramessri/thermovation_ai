"""
vggt_reconstruct.py — feed-forward multi-view reconstruction via VGGT
(facebookresearch/vggt, CVPR 2025 Best Paper), for 9.3's third candidate
alongside MASt3R.

Architecturally different from MASt3R on purpose, not redundantly the same
arm twice: VGGT is a single unified forward pass over ALL input views at
once (no pairwise matching + separate global-alignment optimization step),
reportedly seconds even for hundreds of views — so this arm is really
testing feed-forward reconstruction SPEED and whether a much simpler
architecture still reconstructs the SfM-fragmented section, not the
metric-scale-training question (that's MASt3R's job).

**Scale handling is genuinely different from every other arm here.**
VGGT's own paper and multiple follow-up works confirm its raw output
"inherently lacks metric scale" (unlike MASt3R's dedicated metric
checkpoint) — so this script calibrates it externally: VGGT reconstructs
completely independently (no SfM camera poses fed in, same as MASt3R), but
afterward its own camera-to-camera distances are compared against the
marker-scaled SfM camera trajectory for the SAME frames (matched by
filename) to solve a single scale ratio — same tightest-window robust
aggregation 08_depth/depth_scale.py uses for its own scale ratios, just
applied to camera-pair distances instead of point depths. This still
requires the marker/SfM model to exist for THIS calibration step, unlike
MASt3R — that distinction is itself worth reporting, not glossed over.

Usage:
  python 3d_recons_alternatives/vggt_reconstruct.py \
      --frames-dir dataset/frames/IMG_3126_sfm \
      --sfm-dir output/sfm/IMG_3126 --model 1 \
      --out 3d_recons_alternatives/results/IMG_3126_vggt
  # target the section where classical SfM fragmented:
  python 3d_recons_alternatives/vggt_reconstruct.py \
      --frames-dir dataset/frames/IMG_3126_sfm --frame-range 180:226 \
      --sfm-dir output/sfm/IMG_3126 --model 1 \
      --out 3d_recons_alternatives/results/IMG_3126_vggt_fragment
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pycolmap

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "05_geometry"))
from room_dims import rotation_to_z, density_peak, compute_room_dims, render_topdown


def up_from_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    """Gravity prior from VGGT's own world-to-camera extrinsics.

    VGGT's pose_encoding_to_extri_intri() returns WORLD-TO-CAMERA [R|t]
    matrices — the SAME convention pycolmap's cam_from_world() uses (unlike
    MASt3R's camera-TO-world convention in mast3r_reconstruct.py's
    up_from_poses, which needed the opposite row/column handling). Row 1 of
    the rotation block is the camera's Y axis expressed in world
    coordinates, matching room_dims.py's up_from_cameras exactly."""
    downs = [T[1, :3] for T in extrinsics]
    down = np.mean(downs, axis=0)
    return -down / np.linalg.norm(down)


def camera_centers_from_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    """World-frame camera centers from world-to-camera [R|t]: C = -R^T @ t."""
    return np.stack([-Rt[:, :3].T @ Rt[:, 3] for Rt in extrinsics])


def _robust_mode(ratios: np.ndarray) -> tuple[float, float]:
    """Same 'tightest-window' robust aggregation depth_scale.py uses:
    median of the densest 50% cluster, plus its coefficient of variation."""
    r = np.array(sorted(ratios))
    k = max(int(0.5 * len(r)), 3)
    _, best = min((r[i + k - 1] - r[i], i) for i in range(len(r) - k + 1))
    core = r[best:best + k]
    cm_per_unit = float(np.median(core))
    cv_pct = float(np.std(core) / np.mean(core) * 100)
    return cm_per_unit, cv_pct


def fit_scale_to_sfm(vggt_centers: np.ndarray, vggt_names: list[str],
                     rec: pycolmap.Reconstruction, cm_per_unit: float,
                     max_pairs: int = 500) -> dict | None:
    """Ratio of pairwise camera-center distances: VGGT's arbitrary units vs
    the marker-scaled SfM model's cm, matched by frame name. VGGT never
    sees the marker or the SfM model during its own reconstruction — this
    is purely an external ruler applied afterward to interpret its
    otherwise-unitless output, exactly the calibration step VGGT's "lacks
    metric scale" limitation requires."""
    name_to_sfm_center_cm = {im.name: np.asarray(im.projection_center()) * cm_per_unit
                             for im in rec.images.values()}
    common = [i for i, n in enumerate(vggt_names) if n in name_to_sfm_center_cm]
    if len(common) < 5:
        return None

    rng = np.random.default_rng(0)
    pairs = [(a, b) for ai, a in enumerate(common) for b in common[ai + 1:]]
    if len(pairs) > max_pairs:
        idx = rng.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]

    ratios = []
    for a, b in pairs:
        d_vggt = float(np.linalg.norm(vggt_centers[a] - vggt_centers[b]))
        d_sfm = float(np.linalg.norm(name_to_sfm_center_cm[vggt_names[a]]
                                     - name_to_sfm_center_cm[vggt_names[b]]))
        if d_vggt > 1e-6:
            ratios.append(d_sfm / d_vggt)
    if len(ratios) < 20:
        return None

    cm_per_vggt_unit, cv_pct = _robust_mode(np.array(ratios))
    return dict(cm_per_vggt_unit=cm_per_vggt_unit, cv_pct=round(cv_pct, 2),
               frames_matched=len(common), pairs_used=len(ratios))


def select_frames(frames_dir: Path, frame_range: str | None, every_n: int) -> list[Path]:
    frames = sorted(frames_dir.glob("*.jpg"))
    if frame_range:
        lo, hi = (int(x) for x in frame_range.split(":"))
        frames = frames[lo:hi]
    return frames[::every_n]


def main():
    parser = argparse.ArgumentParser(description="Feed-forward reconstruction via VGGT")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--sfm-dir", type=Path, required=True,
                        help="Existing SfM dir (scale.json + sparse/<model>) — used ONLY to "
                             "calibrate VGGT's arbitrary scale afterward, not fed into VGGT itself")
    parser.add_argument("--model", default="1")
    parser.add_argument("--frame-range", default=None,
                        help="start:end frame INDEX slice — e.g. 180:226 to target the section "
                             "where classical SfM fragmented")
    parser.add_argument("--every-n-frames", type=int, default=2,
                        help="VGGT handles hundreds of views in seconds per its own docs, so this "
                             "can be gentler than MASt3R's subsampling — still configurable")
    parser.add_argument("--conf-threshold", type=float, default=1.0,
                        help="Drop points below this world_points_conf (VGGT's own per-point "
                             "confidence, not a fixed physical unit — tune per video)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    all_frames = sorted(args.frames_dir.glob("*.jpg"))
    frame_paths = select_frames(args.frames_dir, args.frame_range, args.every_n_frames)
    print(f"Using {len(frame_paths)} frames (of {len(all_frames)} available in {args.frames_dir})")
    if len(frame_paths) < 3:
        print("FAILED: need at least 3 frames")
        return

    print("Loading facebook/VGGT-1B...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    t0 = time.perf_counter()
    images = load_and_preprocess_images([str(p) for p in frame_paths]).to(device)
    with torch.no_grad():
        if device == "cuda":
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
        else:
            predictions = model(images)
    runtime_s = time.perf_counter() - t0
    print(f"Reconstruction done in {runtime_s:.1f}s")

    h, w = images.shape[-2:]
    extrinsics, _intrinsics = pose_encoding_to_extri_intri(
        predictions["pose_enc"], image_size_hw=(h, w))
    extrinsics = extrinsics.squeeze(0).float().cpu().numpy()   # (S, 3, 4)

    world_points = predictions["world_points"].squeeze(0).float().cpu().numpy()   # (S, H, W, 3)
    conf = predictions["world_points_conf"].squeeze(0).float().cpu().numpy()      # (S, H, W)
    names = [p.name for p in frame_paths]

    keep = conf > args.conf_threshold
    pts_raw = world_points[keep]
    print(f"Points: {keep.sum()} kept / {conf.size} total (conf > {args.conf_threshold})")
    if len(pts_raw) < 100:
        print("FAILED: too few confident points")
        return

    scale_info = json.loads((args.sfm_dir / "scale.json").read_text())
    cm_per_unit = scale_info["cm_per_unit"]
    rec = pycolmap.Reconstruction(str(args.sfm_dir / "sparse" / args.model))
    cam_centers_raw = camera_centers_from_extrinsics(extrinsics)
    fit = fit_scale_to_sfm(cam_centers_raw, names, rec, cm_per_unit)
    if fit is None:
        print("FAILED: could not fit a scale ratio against the SfM camera trajectory "
              "(too few matching frame names — check --frames-dir matches the SfM run's frames)")
        return
    print(f"Scale calibration: {fit['cm_per_vggt_unit']:.4f} cm per VGGT unit "
          f"(from {fit['frames_matched']} matched frames, {fit['pairs_used']} pairs, "
          f"CV={fit['cv_pct']}%)")

    cm_per_vggt_unit = fit["cm_per_vggt_unit"]
    pts_cm = pts_raw * cm_per_vggt_unit
    cam_centers_cm = cam_centers_raw * cm_per_vggt_unit

    up = up_from_extrinsics(extrinsics)
    R = rotation_to_z(up)
    z_all = (pts_cm @ R.T)[:, 2]
    z_floor = density_peak(z_all, 0.5, 35.0)
    if z_floor is None:
        print("FAILED: could not localize the floor density peak")
        return

    metrics, geometry = compute_room_dims(pts_cm, R, z_floor, cam_centers_cm)
    bound = ">= " if metrics["height_is_lower_bound"] else ""
    print(f"\nRoom height : {bound}{metrics['height_cm']:.1f} cm"
          f"{'  [LOW CONFIDENCE]' if not metrics['height_reliable'] else ''}")
    print(f"Room length : {metrics['length_cm']:.1f} cm")
    print(f"Room width  : {metrics['width_cm']:.1f} cm")

    args.out.mkdir(parents=True, exist_ok=True)
    viz_path = args.out / "vggt_room_topdown.png"
    render_topdown(geometry, R, z_floor, cm_per_vggt_unit, cam_centers_cm, None, viz_path)

    out = dict(source="vggt", checkpoint="facebook/VGGT-1B",
              frames_used=len(frame_paths), frames_available=len(all_frames),
              frame_range=args.frame_range, runtime_s=round(runtime_s, 1),
              n_points=int(len(pts_raw)), conf_threshold=args.conf_threshold,
              scale_source="camera_trajectory_ratio_vs_sfm_marker_scale",
              cm_per_vggt_unit=cm_per_vggt_unit, scale_fit_cv_pct=fit["cv_pct"],
              scale_fit_frames_matched=fit["frames_matched"],
              **metrics)
    out_path = args.out / "vggt_room_dims.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path}")
    print(f"Saved {viz_path}")


if __name__ == "__main__":
    main()
