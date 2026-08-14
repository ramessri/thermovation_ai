"""
compare_reconstructions.py — SfM vs 3D Gaussian Splatting vs NeRF vs MASt3R
vs VGGT, side by side, for one video: room dimensions, coverage (points
used), runtime, and optional ground-truth error if you have tape
measurements for this video.

Reads the JSON files each arm already produces (room_dims.json from
05_geometry/room_dims.py, gaussian_room_dims.json from
gaussians_to_room_dims.py, nerf_room_dims.json from
nerf_pointcloud_to_room_dims.py, mast3r_room_dims.json from
mast3r_reconstruct.py, vggt_room_dims.json from vggt_reconstruct.py) — this
script does no reconstruction itself, it only compares numbers that already
exist on disk. 3DGS, NeRF, MASt3R, and VGGT inputs are all optional — pass
whichever arms you've actually run.

Usage:
  python 3d_recons_alternatives/compare_reconstructions.py \
      --video IMG_3126 \
      --sfm-room-dims output/sfm/IMG_3126/room_dims.json \
      --gaussian-room-dims 3d_recons_alternatives/results/IMG_3126/gaussian_room_dims.json \
      --nerf-room-dims 3d_recons_alternatives/results/IMG_3126_nerf/nerf_room_dims.json \
      --mast3r-room-dims 3d_recons_alternatives/results/IMG_3126_mast3r/mast3r_room_dims.json \
      --vggt-room-dims 3d_recons_alternatives/results/IMG_3126_vggt/vggt_room_dims.json \
      --sfm-runtime-s 183 --gsplat-runtime-s 1200 --nerf-runtime-s 900 \
      --gt-length-cm 525 --gt-width-cm 152
"""

import argparse
import json
from pathlib import Path


def pct_error(value: float | None, gt: float | None) -> float | None:
    if value is None or gt is None or gt == 0:
        return None
    return round((value - gt) / gt * 100, 1)


def fmt(v) -> str:
    return f"{v:.1f}" if isinstance(v, (int, float)) else "-"


def main():
    parser = argparse.ArgumentParser(description="Compare SfM vs 3DGS vs NeRF room-dimension results")
    parser.add_argument("--video", required=True)
    parser.add_argument("--sfm-room-dims", type=Path, required=True)
    parser.add_argument("--gaussian-room-dims", type=Path, default=None)
    parser.add_argument("--nerf-room-dims", type=Path, default=None)
    parser.add_argument("--mast3r-room-dims", type=Path, default=None)
    parser.add_argument("--vggt-room-dims", type=Path, default=None)
    parser.add_argument("--sfm-runtime-s", type=float, default=None,
                        help="From run.py's printed summary for this video")
    parser.add_argument("--gsplat-runtime-s", type=float, default=None,
                        help="From gsplat's own training-summary output — not captured "
                             "automatically, note it from the console and pass it here")
    parser.add_argument("--nerf-runtime-s", type=float, default=None,
                        help="From ns-train's own printed summary — not captured "
                             "automatically, note it from the console and pass it here")
    parser.add_argument("--mast3r-runtime-s", type=float, default=None,
                        help="Optional override — mast3r_reconstruct.py already saves its own "
                             "runtime_s in mast3r_room_dims.json, used automatically otherwise")
    parser.add_argument("--vggt-runtime-s", type=float, default=None,
                        help="Optional override — vggt_reconstruct.py already saves its own "
                             "runtime_s in vggt_room_dims.json, used automatically otherwise")
    parser.add_argument("--gt-length-cm", type=float, default=None)
    parser.add_argument("--gt-width-cm", type=float, default=None)
    parser.add_argument("--gt-height-cm", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: 3d_recons_alternatives/results/<video>/comparison.json")
    args = parser.parse_args()

    arms = {"sfm": json.loads(args.sfm_room_dims.read_text())}
    runtimes = {"sfm": args.sfm_runtime_s}
    if args.gaussian_room_dims:
        arms["gsplat"] = json.loads(args.gaussian_room_dims.read_text())
        runtimes["gsplat"] = args.gsplat_runtime_s
    if args.nerf_room_dims:
        arms["nerf"] = json.loads(args.nerf_room_dims.read_text())
        runtimes["nerf"] = args.nerf_runtime_s
    if args.mast3r_room_dims:
        arms["mast3r"] = json.loads(args.mast3r_room_dims.read_text())
        runtimes["mast3r"] = args.mast3r_runtime_s or arms["mast3r"].get("runtime_s")
    if args.vggt_room_dims:
        arms["vggt"] = json.loads(args.vggt_room_dims.read_text())
        runtimes["vggt"] = args.vggt_runtime_s or arms["vggt"].get("runtime_s")

    gt = dict(length_cm=args.gt_length_cm, width_cm=args.gt_width_cm, height_cm=args.gt_height_cm)

    rows = []
    for dim in ("length_cm", "width_cm", "height_cm"):
        row = dict(dim=dim, gt=gt[dim])
        for arm_name, data in arms.items():
            row[arm_name] = data.get(dim)
            row[f"{arm_name}_err_pct"] = pct_error(data.get(dim), gt[dim])
        rows.append(row)

    report = dict(video=args.video, arms_compared=list(arms.keys()), room_dims=rows)
    for arm_name, data in arms.items():
        report[arm_name] = dict(
            reliable_height=data.get("height_reliable"),
            reliable_footprint=data.get("footprint_reliable"),
            points_used=data.get("points_used"),
            runtime_s=runtimes.get(arm_name),
        )
        if arm_name == "gsplat":
            report[arm_name]["n_gaussians_total"] = data.get("n_gaussians_total")
            report[arm_name]["n_gaussians_kept"] = data.get("n_gaussians_kept")
        if arm_name == "nerf":
            report[arm_name]["world_frame_sanity_offset_cm"] = data.get("world_frame_sanity_offset_cm")
        if arm_name in ("mast3r", "vggt"):
            report[arm_name]["scale_source"] = data.get("scale_source")
            report[arm_name]["frames_used"] = data.get("frames_used")
            report[arm_name]["frame_range"] = data.get("frame_range")
        if arm_name == "vggt":
            report[arm_name]["scale_fit_cv_pct"] = data.get("scale_fit_cv_pct")

    print(f"=== {args.video}: {' vs '.join(a.upper() for a in arms)} ===\n")
    header = f"{'dim':<12}"
    for arm_name in arms:
        header += f"{arm_name:>12}{'err%':>8}"
    header += f"{'GT':>10}"
    print(header)
    for r in rows:
        line = f"{r['dim']:<12}"
        for arm_name in arms:
            line += f"{fmt(r[arm_name]):>12}{fmt(r[f'{arm_name}_err_pct']):>8}"
        line += f"{fmt(r['gt']):>10}"
        print(line)

    print()
    for arm_name, data in arms.items():
        extra = ""
        if arm_name == "gsplat":
            extra = f" ({data.get('n_gaussians_kept')}/{data.get('n_gaussians_total')} Gaussians kept)"
        print(f"Coverage [{arm_name}]: {data.get('points_used')} points{extra}  "
              f"(reliable: height={data.get('height_reliable')}, "
              f"footprint={data.get('footprint_reliable')})")
        if arm_name == "nerf" and data.get("world_frame_sanity_offset_cm") is not None:
            print(f"  world-frame sanity offset: {data['world_frame_sanity_offset_cm']:.0f}cm "
                  "(large -> --save-world-frame export may not be trustworthy)")
        if arm_name in ("mast3r", "vggt"):
            cv_note = f", scale-fit CV={data['scale_fit_cv_pct']}%" if data.get("scale_fit_cv_pct") is not None else ""
            print(f"  scale source: {data.get('scale_source')}{cv_note}"
                  f"{' (frames ' + str(data.get('frame_range')) + ')' if data.get('frame_range') else ''}")

    known_runtimes = {k: v for k, v in runtimes.items() if v}
    if len(known_runtimes) > 1:
        print()
        base = known_runtimes.get("sfm")
        for arm_name, t in known_runtimes.items():
            ratio = f" ({t / base:.1f}x SfM)" if base and arm_name != "sfm" else ""
            print(f"Runtime [{arm_name}]: {t:.0f}s{ratio}")

    out_path = args.output or Path(f"3d_recons_alternatives/results/{args.video}/comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
