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

Usage (single video):
  python 3d_recons_alternatives/vggt_reconstruct.py \
      --frames-dir dataset/frames/IMG_3126_sfm \
      --sfm-dir output/sfm/IMG_3126 --model 1 \
      --out 3d_recons_alternatives/results/IMG_3126_vggt
  # target the section where classical SfM fragmented:
  python 3d_recons_alternatives/vggt_reconstruct.py \
      --frames-dir dataset/frames/IMG_3126_sfm --frame-range 180:226 \
      --sfm-dir output/sfm/IMG_3126 --model 1 \
      --out 3d_recons_alternatives/results/IMG_3126_vggt_fragment

Usage (batch — every video with a completed sfm/ under a root):
  python 3d_recons_alternatives/vggt_reconstruct.py \
      --sfm-root "C:\\thermovation-output\\sfm 1" \
      --out "C:\\thermovation-output\\3d_recons_alternatives\\vggt"
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


def select_frames(frames_dir: Path, frame_range: str | None, every_n: int,
                  max_frames: int | None = None) -> list[Path]:
    frames = sorted(frames_dir.glob("*.jpg"))
    if frame_range:
        lo, hi = (int(x) for x in frame_range.split(":"))
        frames = frames[lo:hi]
    frames = frames[::every_n]
    if max_frames and len(frames) > max_frames:
        # VGGT is a single unified forward pass over every frame's tokens at
        # once — attention cost scales far worse than linear with frame
        # count in practice (measured: 148 frames -> 757s on this GPU, not
        # the "seconds even for hundreds of views" the model card implies).
        # Cap and re-space evenly rather than let a long video silently
        # blow up runtime/VRAM.
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in sorted(set(idx.tolist()))]
    return frames


def write_ply(path: Path, xyz_cm: np.ndarray, rgb: np.ndarray | None, max_points: int = 2_000_000) -> None:
    """Minimal binary PLY writer for the point-cloud viewer. Subsamples if huge."""
    n = len(xyz_cm)
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        xyz_cm, rgb = xyz_cm[idx], (rgb[idx] if rgb is not None else None)
        n = max_points
    if rgb is None:
        rgb = np.full((n, 3), 180, dtype=np.uint8)
    with open(path, "wb") as f:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        ).encode("ascii")
        f.write(header)
        rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                 ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec["x"], rec["y"], rec["z"] = xyz_cm[:, 0], xyz_cm[:, 1], xyz_cm[:, 2]
        rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        f.write(rec.tobytes())


def process_video(frames_dir: Path, sfm_dir: Path, model_name: str, frame_range, every_n_frames: int,
                  conf_threshold: float, model, device: str, dtype, out_dir: Path,
                  max_frames: int | None = None) -> dict:
    import torch
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    scale_path = sfm_dir / "scale.json"
    if not scale_path.exists():
        return dict(error="no scale.json")

    all_frames = sorted(frames_dir.glob("*.jpg"))
    frame_paths = select_frames(frames_dir, frame_range, every_n_frames, max_frames)
    print(f"  Using {len(frame_paths)} frames (of {len(all_frames)} available)", flush=True)
    if len(frame_paths) < 3:
        return dict(error="need at least 3 frames")

    t0 = time.perf_counter()
    images = load_and_preprocess_images([str(p) for p in frame_paths]).to(device)
    with torch.no_grad():
        if device == "cuda":
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
        else:
            predictions = model(images)
    runtime_s = time.perf_counter() - t0
    print(f"  Reconstruction done in {runtime_s:.1f}s", flush=True)

    h, w = images.shape[-2:]
    extrinsics, _intrinsics = pose_encoding_to_extri_intri(
        predictions["pose_enc"], image_size_hw=(h, w))
    extrinsics = extrinsics.squeeze(0).float().cpu().numpy()   # (S, 3, 4)

    world_points = predictions["world_points"].squeeze(0).float().cpu().numpy()   # (S, H, W, 3)
    conf = predictions["world_points_conf"].squeeze(0).float().cpu().numpy()      # (S, H, W)
    names = [p.name for p in frame_paths]

    # best-effort per-point color for the viewer: images is (S,3,H,W), 0-1 range
    imgs_np = (images.squeeze(0) if images.dim() == 5 else images).float().cpu().numpy()
    rgb_full = np.clip(imgs_np.transpose(0, 2, 3, 1) * 255.0, 0, 255).astype(np.uint8)  # (S,H,W,3)

    keep = conf > conf_threshold
    pts_raw = world_points[keep]
    rgb_kept = rgb_full[keep] if rgb_full.shape[:3] == keep.shape else None
    print(f"  Points: {keep.sum()} kept / {conf.size} total (conf > {conf_threshold})", flush=True)
    if len(pts_raw) < 100:
        return dict(error="too few confident points")

    scale_info = json.loads(scale_path.read_text())
    cm_per_unit = scale_info["cm_per_unit"]
    rec = pycolmap.Reconstruction(str(sfm_dir / "sparse" / model_name))
    cam_centers_raw = camera_centers_from_extrinsics(extrinsics)
    fit = fit_scale_to_sfm(cam_centers_raw, names, rec, cm_per_unit)
    if fit is None:
        return dict(error="could not fit a scale ratio against the SfM camera trajectory")
    print(f"  Scale calibration: {fit['cm_per_vggt_unit']:.4f} cm per VGGT unit "
          f"(CV={fit['cv_pct']}%)", flush=True)

    cm_per_vggt_unit = fit["cm_per_vggt_unit"]
    pts_cm = pts_raw * cm_per_vggt_unit
    cam_centers_cm = cam_centers_raw * cm_per_vggt_unit

    up = up_from_extrinsics(extrinsics)
    R = rotation_to_z(up)
    z_all = (pts_cm @ R.T)[:, 2]
    z_floor = density_peak(z_all, 0.5, 35.0)
    if z_floor is None:
        return dict(error="could not localize the floor density peak")

    metrics, geometry = compute_room_dims(pts_cm, R, z_floor, cam_centers_cm)
    bound = ">= " if metrics["height_is_lower_bound"] else ""
    print(f"  Room height : {bound}{metrics['height_cm']:.1f} cm"
          f"{'  [LOW CONFIDENCE]' if not metrics['height_reliable'] else ''}", flush=True)
    print(f"  Room length : {metrics['length_cm']:.1f} cm", flush=True)
    print(f"  Room width  : {metrics['width_cm']:.1f} cm", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    viz_path = out_dir / "vggt_room_topdown.png"
    render_topdown(geometry, R, z_floor, cm_per_vggt_unit, cam_centers_cm, None, viz_path)

    ply_path = out_dir / "vggt_pointcloud.ply"
    write_ply(ply_path, pts_cm, rgb_kept)

    result = dict(source="vggt", checkpoint="facebook/VGGT-1B",
                 frames_used=len(frame_paths), frames_available=len(all_frames),
                 frame_range=frame_range, runtime_s=round(runtime_s, 1),
                 n_points=int(len(pts_raw)), conf_threshold=conf_threshold,
                 scale_source="camera_trajectory_ratio_vs_sfm_marker_scale",
                 cm_per_vggt_unit=cm_per_vggt_unit, scale_fit_cv_pct=fit["cv_pct"],
                 scale_fit_frames_matched=fit["frames_matched"],
                 **metrics)
    out_path = out_dir / "vggt_room_dims.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  Saved {out_path}", flush=True)
    print(f"  Saved {ply_path}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="Feed-forward reconstruction via VGGT")
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--sfm-dir", type=Path,
                        help="Existing SfM dir (scale.json + sparse/<model>) — used ONLY to "
                             "calibrate VGGT's arbitrary scale afterward, not fed into VGGT itself")
    parser.add_argument("--sfm-root", type=Path,
                        help="Batch mode: root containing <video>/sfm + <video>/frames subfolders")
    parser.add_argument("--model", default=None,
                        help="Sparse model subfolder name (default: auto-pick most-registered)")
    parser.add_argument("--frame-range", default=None,
                        help="start:end frame INDEX slice — e.g. 180:226 to target the section "
                             "where classical SfM fragmented")
    parser.add_argument("--every-n-frames", type=int, default=2,
                        help="VGGT handles hundreds of views in seconds per its own docs, so this "
                             "can be gentler than MASt3R's subsampling — still configurable")
    parser.add_argument("--max-frames", type=int, default=60,
                        help="Hard cap regardless of --every-n-frames — measured 148 frames at "
                             "~757s on this GPU (attention scales far worse than linear in "
                             "practice), so long videos need a ceiling, not just a stride")
    parser.add_argument("--conf-threshold", type=float, default=1.0,
                        help="Drop points below this world_points_conf (VGGT's own per-point "
                             "confidence, not a fixed physical unit — tune per video)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true",
                        help="Batch mode: redo videos that already have output (default: skip/resume)")
    args = parser.parse_args()
    if not args.frames_dir and not args.sfm_root:
        parser.error("pass --frames-dir/--sfm-dir (single video) or --sfm-root (batch)")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

    from vggt.models.vggt import VGGT
    print("Loading facebook/VGGT-1B...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    if args.frames_dir:
        jobs = [(args.frames_dir, args.sfm_dir, args.out)]
    else:
        jobs = sorted(
            (p.parent / "frames", p.parent / "sfm", args.out / p.parent.name)
            for p in args.sfm_root.glob("*/sfm") if p.is_dir()
        )
        print(f"Batch mode: {len(jobs)} videos found under {args.sfm_root}")

    def best_sparse_model(sfm_dir: Path) -> str | None:
        best, best_n = None, -1
        sparse = sfm_dir / "sparse"
        if not sparse.exists():
            return None
        for model_dir in sorted(sparse.iterdir()):
            if not (model_dir / "images.bin").exists():
                continue
            rec = pycolmap.Reconstruction(str(model_dir))
            if rec.num_reg_images() > best_n:
                best, best_n = model_dir.name, rec.num_reg_images()
        return best

    summary = []
    for frames_dir, sfm_dir, out_dir in jobs:
        video_name = out_dir.name
        existing_path = out_dir / "vggt_room_dims.json"
        if args.sfm_root and not args.force and existing_path.exists():
            result = json.loads(existing_path.read_text())
            result["video"] = video_name
            summary.append(result)
            bound = ">=" if result.get("height_is_lower_bound") else ""
            print(f"\n=== {video_name} === (skipped, already done: "
                  f"H={bound}{result.get('height_cm', 0):.1f}cm)", flush=True)
            continue

        print(f"\n=== {video_name} ===", flush=True)
        model_name = args.model or best_sparse_model(sfm_dir)
        if model_name is None:
            result = dict(error="no sparse model found")
        else:
            try:
                result = process_video(frames_dir, sfm_dir, model_name, args.frame_range,
                                       args.every_n_frames, args.conf_threshold, model, device,
                                       dtype, out_dir, args.max_frames)
            except Exception as exc:
                result = dict(error=f"{type(exc).__name__}: {exc}")
            if device == "cuda":
                torch.cuda.empty_cache()
        result["video"] = video_name
        summary.append(result)
        if "error" in result:
            print(f"  FAILED: {result['error']}", flush=True)

    if args.sfm_root:
        summary_path = args.out / "summary.json"
        args.out.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\nSaved batch summary: {summary_path}")
        print("\n" + "=" * 72)
        print(f"{'video':<45} {'H(cm)':>8} {'L(cm)':>8} {'W(cm)':>8}  runtime(s)")
        for r in summary:
            if "error" in r:
                print(f"{r['video']:<45} FAILED: {r['error']}")
            else:
                print(f"{r['video']:<45} {r['height_cm']:>8.0f} {r['length_cm']:>8.0f} "
                      f"{r['width_cm']:>8.0f}  {r.get('runtime_s', '?')}")


if __name__ == "__main__":
    main()
