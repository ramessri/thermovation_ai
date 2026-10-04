"""
mast3r_reconstruct.py — feed-forward multi-view reconstruction via MASt3R
(no training, no seeding from the existing SfM run — this arm gets nothing
from the rest of the pipeline except the raw video frames), for 9.3.

Three questions this answers directly:
  1. Runtime vs classical SfM, on the same frames (printed + saved).
  2. Does it reconstruct the section of a video where classical SfM
     fragmented? Pass --frame-range to target that section specifically
     (IMG_3126's own SfM run fragments into a 194-image main model + a
     41-image fragment — see the main README's SfM baseline section for
     which frame indices that corresponds to).
  3. **The headline question**: does MASt3R's own metric-scale training
     make its point cloud usably metric WITHOUT the marker at all? Every
     other arm in this project (SfM, 3DGS, NeRF, the 08_depth/ scale
     methods) still needs cm_per_unit from marker/Zhang calibration to turn
     its point cloud into centimeters. This arm never touches scale.json —
     the metric checkpoint's output is used AS METRIC directly (just a
     unit conversion, meters -> cm), and that's the actual thing being
     tested: is that number trustworthy?

Requires the naver/mast3r repo installed (not just the model weights) — see
3d_recons_alternatives/README.md section D0 for setup. Three real bugs were
found and fixed getting this to actually run (2026-08-20, verified against
the installed repo, not just its public source as originally written):
  1. The vendored mast3r/cloud_opt/sparse_ga.py itself calls
     `scipy.cluster.hierarchy.distance.squareform(...)`, which doesn't exist
     on scipy>=1.18 (older scipy versions implicitly exposed `.distance`
     as a side effect of internal imports; that stopped working). Patched
     the vendored file directly to import squareform from scipy.spatial.
  2. `get_dense_pts3d()` returns 3 values (pts3d, depthmaps, confs), not the
     2 this script originally unpacked.
  3. The returned pts3d tensors are still on the GPU — need .cpu() before
     .numpy().

Usage (single video):
  python 3d_recons_alternatives/mast3r_reconstruct.py \
      --frames-dir dataset/frames/IMG_3126_sfm \
      --out 3d_recons_alternatives/results/IMG_3126_mast3r
  # target the specific frame range where classical SfM fragmented:
  python 3d_recons_alternatives/mast3r_reconstruct.py \
      --frames-dir dataset/frames/IMG_3126_sfm --frame-range 180:226 \
      --out 3d_recons_alternatives/results/IMG_3126_mast3r_fragment

Usage (batch — every "<video>/frames" folder under a root):
  python 3d_recons_alternatives/mast3r_reconstruct.py \
      --frames-root "C:\\thermovation-output\\sfm 1" \
      --out "C:\\thermovation-output\\3d_recons_alternatives\\mast3r"
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "05_geometry"))
from room_dims import rotation_to_z, density_peak, compute_room_dims, render_topdown

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def up_from_poses(im_poses: np.ndarray) -> np.ndarray:
    """Gravity prior from MASt3R's own camera-to-world poses.

    MASt3R/dust3r's get_im_poses() returns camera-TO-world (world_from_cam)
    4x4 matrices — the OPPOSITE convention from pycolmap's cam_from_world()
    used everywhere else in this project. A camera's local Y-axis ("down"
    in typical image conventions) expressed in world coordinates is the
    SECOND COLUMN of the rotation block here (not a row, as it would be for
    a world->cam matrix in room_dims.py's up_from_cameras) — getting this
    backwards silently produces a plausible-looking but wrong gravity
    direction, so it's called out explicitly rather than copy-pasted.
    """
    downs = [T[:3, 1] for T in im_poses]
    down = np.mean(downs, axis=0)
    return -down / np.linalg.norm(down)


def select_frames(frames_dir: Path, frame_range: str | None, every_n: int) -> list[Path]:
    frames = sorted(frames_dir.glob("*.jpg"))
    if frame_range:
        lo, hi = (int(x) for x in frame_range.split(":"))
        frames = frames[lo:hi]
    return frames[::every_n]


def write_ply(path: Path, xyz_cm: np.ndarray, rgb: np.ndarray | None, max_points: int = 2_000_000) -> None:
    """Minimal ASCII-header/binary-body PLY writer for the point-cloud viewer.
    Subsamples if huge — a browser viewer doesn't need 3M+ raw points."""
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


def process_video(frames_dir: Path, out_dir: Path, frame_range, every_n_frames: int,
                  win_size: int, checkpoint: str, model, device: str) -> dict:
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    from dust3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images

    all_frames = sorted(frames_dir.glob("*.jpg"))
    frame_paths = select_frames(frames_dir, frame_range, every_n_frames)
    print(f"  Using {len(frame_paths)} frames (of {len(all_frames)} available)", flush=True)
    if len(frame_paths) < 3:
        return dict(error="need at least 3 frames")

    t0 = time.perf_counter()
    imgs = load_images([str(p) for p in frame_paths], size=512, verbose=False)
    pairs = make_pairs(imgs, scene_graph=f"swin-{win_size}", prefilter=None, symmetrize=True)
    print(f"  {len(pairs)} pairs from {len(imgs)} images", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    scene = sparse_global_alignment(
        [str(p) for p in frame_paths], pairs, str(out_dir / "cache"), model, device=device,
        verbose=False,
    )
    runtime_s = time.perf_counter() - t0
    print(f"  Reconstruction done in {runtime_s:.1f}s", flush=True)

    pts3d_list, _depthmaps, _confs = scene.get_dense_pts3d(clean_depth=True)
    pts_m_list = [p.detach().cpu().numpy().reshape(-1, 3) for p in pts3d_list]
    pts_m = np.concatenate(pts_m_list, axis=0)
    im_poses = np.asarray(scene.get_im_poses().detach().cpu())
    focals = np.asarray(scene.get_focals().detach().cpu())
    print(f"  Dense point cloud: {len(pts_m)} points, {len(im_poses)} camera poses", flush=True)

    # best-effort per-point color for the viewer: denormalize each view's
    # processed (512-resized) image tensor back to 0-255 RGB
    rgb_list = []
    for i, img_dict in enumerate(imgs):
        t = img_dict["img"][0].permute(1, 2, 0).cpu().numpy()  # (H,W,3), ImageNet-normalized
        rgb_i = np.clip((t * IMAGENET_STD + IMAGENET_MEAN) * 255.0, 0, 255).astype(np.uint8)
        rgb_list.append(rgb_i.reshape(-1, 3))
    rgb = np.concatenate(rgb_list, axis=0) if len(rgb_list) == len(pts_m_list) else None

    # MASt3R's metric-scale checkpoint outputs points already in METERS —
    # this is the actual test of "does metric-scale training remove marker
    # dependency": no scale.json, no cm_per_unit ratio-fitting, just x100.
    pts_cm = pts_m * 100.0
    cam_centers_cm = im_poses[:, :3, 3] * 100.0

    up = up_from_poses(im_poses)
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

    viz_path = out_dir / "mast3r_room_topdown.png"
    render_topdown(geometry, R, z_floor, 1.0, cam_centers_cm, None, viz_path)

    ply_path = out_dir / "mast3r_pointcloud.ply"
    write_ply(ply_path, pts_cm, rgb)

    result = dict(source="mast3r", checkpoint=checkpoint,
                 frames_used=len(frame_paths), frames_available=len(all_frames),
                 frame_range=frame_range, runtime_s=round(runtime_s, 1),
                 n_points=len(pts_m),
                 scale_source="mast3r_native_metric (no marker, no scale.json)",
                 **metrics)
    out_path = out_dir / "mast3r_room_dims.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  Saved {out_path}", flush=True)
    print(f"  Saved {ply_path}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="Feed-forward reconstruction via MASt3R")
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--frames-root", type=Path,
                        help="Batch mode: root containing <video>/frames subfolders")
    parser.add_argument("--frame-range", default=None,
                        help="start:end frame INDEX slice (post-sort, pre-subsample) — e.g. "
                             "180:226 to target the specific section where classical SfM fragmented")
    parser.add_argument("--every-n-frames", type=int, default=3,
                        help="Pairwise inference is heavier than SfM's feature matching even with "
                             "windowed pairing — subsample")
    parser.add_argument("--win-size", type=int, default=3,
                        help="Sliding-window pair radius for scene_graph='swin-N' (video-appropriate, "
                             "not all-pairs) — each frame pairs with its N nearest neighbors")
    parser.add_argument("--checkpoint", default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true",
                        help="Batch mode: redo videos that already have output (default: skip/resume)")
    args = parser.parse_args()
    if not args.frames_dir and not args.frames_root:
        parser.error("pass --frames-dir (single video) or --frames-root (batch)")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from mast3r.model import AsymmetricMASt3R
    print(f"Loading {args.checkpoint}...")
    model = AsymmetricMASt3R.from_pretrained(args.checkpoint).to(device)

    if args.frames_dir:
        jobs = [(args.frames_dir, args.out)]
    else:
        jobs = sorted(
            (p, args.out / p.parent.name)
            for p in args.frames_root.glob("*/frames") if p.is_dir()
        )
        print(f"Batch mode: {len(jobs)} videos found under {args.frames_root}")

    summary = []
    for frames_dir, out_dir in jobs:
        video_name = out_dir.name
        existing_path = out_dir / "mast3r_room_dims.json"
        if args.frames_root and not args.force and existing_path.exists():
            result = json.loads(existing_path.read_text())
            result["video"] = video_name
            summary.append(result)
            bound = ">=" if result.get("height_is_lower_bound") else ""
            print(f"\n=== {video_name} === (skipped, already done: "
                  f"H={bound}{result.get('height_cm', 0):.1f}cm)", flush=True)
            continue

        print(f"\n=== {video_name} ===", flush=True)
        try:
            result = process_video(frames_dir, out_dir, args.frame_range, args.every_n_frames,
                                   args.win_size, args.checkpoint, model, device)
        except Exception as exc:
            result = dict(error=f"{type(exc).__name__}: {exc}")
        if device == "cuda":
            torch.cuda.empty_cache()
        result["video"] = video_name
        summary.append(result)
        if "error" in result:
            print(f"  FAILED: {result['error']}", flush=True)

    if args.frames_root:
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
