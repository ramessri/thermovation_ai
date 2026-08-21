"""
run_nerf_gsplat_batch.py — batch-runs BOTH NeRF (nerfacto, torch implementation)
and 3D Gaussian Splatting (gsplat, CUDA rasterizer via portable-MSVC) over every
video in the "sfm 1" calibrated batch, seeded from each video's existing SfM
sparse model + scale.json (same pattern as mast3r_reconstruct.py / vggt_reconstruct.py).

Neither method needs admin rights on this machine:
  - NeRF's torch implementation avoids tinycudann (no compiler needed at all),
    just ffmpeg + COLMAP CLI binaries on PATH (portable downloads, see README).
  - gsplat's CUDA rasterizer DOES need a real C++ compiler at JIT-compile time;
    that's provided by a portable MSVC + Windows SDK downloaded with
    3d_recons_alternatives/../portable-msvc (mmozeiko's script) into
    C:\\thermovation-repos\\portable-msvc\\msvc — no installer, no admin.
    CUDA 13.2's headers additionally require NVCC_APPEND_FLAGS=-Xcompiler
    /Zc:preprocessor or nvcc's C1189 fatal error fires on the traditional
    MSVC preprocessor.

Iteration counts (NERF_ITERS / GSPLAT_STEPS below) are deliberately small —
this is a fast coverage-first pass across all videos, not a per-video quality
run. See README for the reasoning (measured ~4.8 it/s for NeRF-torch and
~17.5 it/s for gsplat on this GPU once JIT-warm).

Every stage is independently resumable (skipped if its output already exists)
and wrapped in try/except so one video's failure doesn't stop the batch. A
wall-clock DEADLINE_HOURS cutoff stops starting new videos once time runs out,
so partial coverage degrades gracefully instead of the batch overrunning.

Usage:
  python 3d_recons_alternatives/run_nerf_gsplat_batch.py
  python 3d_recons_alternatives/run_nerf_gsplat_batch.py --only IMG_3126
  python 3d_recons_alternatives/run_nerf_gsplat_batch.py --skip-nerf
  python 3d_recons_alternatives/run_nerf_gsplat_batch.py --skip-gsplat
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pycolmap

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

SFM_ROOT = Path(r"C:\thermovation-output\sfm 1")
OUT = Path(r"C:\thermovation-output\3d_recons_alternatives")
NERF_DATA = OUT / "nerf_data"
GSPLAT_DATA = OUT / "gsplat_data"
RESULTS = OUT / "results"
GSPLAT_RESULTS = OUT / "gsplat_results"

FFMPEG_BIN = r"C:\thermovation-repos\ffmpeg\ffmpeg-master-latest-win64-gpl\bin"
COLMAP_BIN = r"C:\thermovation-repos\colmap\bin"
MSVC_ROOT = r"C:\thermovation-repos\portable-msvc\msvc"
MSVC_VER = "14.51.36231"
SDK_VER = "10.0.28000.0"
GSPLAT_EXAMPLES = Path(r"C:\thermovation-repos\gsplat\examples")

NERF_ITERS = 500
GSPLAT_STEPS = 1500

DEADLINE_HOURS = 3.75  # leave buffer under the user's 4h check-in
START_TIME = time.time()


def deadline_hit() -> bool:
    return (time.time() - START_TIME) > DEADLINE_HOURS * 3600


def nerf_env() -> dict:
    env = os.environ.copy()
    env["PATH"] = f"{FFMPEG_BIN};{COLMAP_BIN};" + env.get("PATH", "")
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def gsplat_env() -> dict:
    env = os.environ.copy()
    vc_bin = rf"{MSVC_ROOT}\VC\Tools\MSVC\{MSVC_VER}\bin\Hostx64\x64"
    sdk_bin = rf"{MSVC_ROOT}\Windows Kits\10\bin\{SDK_VER}\x64"
    sdk_bin_ucrt = rf"{MSVC_ROOT}\Windows Kits\10\bin\{SDK_VER}\x64\ucrt"
    env["PATH"] = f"{vc_bin};{sdk_bin};{sdk_bin_ucrt};" + env.get("PATH", "")
    inc = [
        rf"{MSVC_ROOT}\VC\Tools\MSVC\{MSVC_VER}\include",
        rf"{MSVC_ROOT}\Windows Kits\10\Include\{SDK_VER}\ucrt",
        rf"{MSVC_ROOT}\Windows Kits\10\Include\{SDK_VER}\shared",
        rf"{MSVC_ROOT}\Windows Kits\10\Include\{SDK_VER}\um",
        rf"{MSVC_ROOT}\Windows Kits\10\Include\{SDK_VER}\winrt",
        rf"{MSVC_ROOT}\Windows Kits\10\Include\{SDK_VER}\cppwinrt",
    ]
    lib = [
        rf"{MSVC_ROOT}\VC\Tools\MSVC\{MSVC_VER}\lib\x64",
        rf"{MSVC_ROOT}\Windows Kits\10\Lib\{SDK_VER}\ucrt\x64",
        rf"{MSVC_ROOT}\Windows Kits\10\Lib\{SDK_VER}\um\x64",
    ]
    env["INCLUDE"] = ";".join(inc)
    env["LIB"] = ";".join(lib)
    env["NVCC_APPEND_FLAGS"] = "-Xcompiler /Zc:preprocessor"
    # setuptools' _get_vc_env() otherwise ignores our INCLUDE/LIB/PATH and
    # tries to rediscover MSVC itself via vswhere/registry, which fails since
    # this portable install isn't a registered Visual Studio instance. These
    # two vars tell distutils/setuptools "trust the environment as-is".
    env["DISTUTILS_USE_SDK"] = "1"
    env["MSSdk"] = "1"
    return env


def run(cmd, label, cwd=None, env=None, timeout=None) -> bool:
    cmd = [str(c) for c in cmd]
    print(f"\n=== {label} ===\n$ {' '.join(cmd)}", flush=True)
    try:
        res = subprocess.run(cmd, cwd=cwd, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  [{label}] TIMED OUT after {timeout}s — continuing", flush=True)
        return False
    except Exception as e:
        print(f"  [{label}] EXCEPTION: {e} — continuing", flush=True)
        return False
    if res.returncode != 0:
        print(f"  [{label}] FAILED (exit {res.returncode}) — continuing", flush=True)
        return False
    return True


def best_sparse_model(sfm_dir: Path):
    sparse = sfm_dir / "sparse"
    if not sparse.exists():
        return None
    best, best_n = None, -1
    for model_dir in sorted(sparse.iterdir()):
        if not (model_dir / "images.bin").exists():
            continue
        rec = pycolmap.Reconstruction(str(model_dir))
        if rec.num_reg_images() > best_n:
            best, best_n = model_dir.name, rec.num_reg_images()
    return best


def do_nerf(video: str, sfm_dir: Path, frames_dir: Path, model: str) -> None:
    data_dir = NERF_DATA / video
    result_dir = RESULTS / video / "nerf"
    export_dir = RESULTS / video / "nerf_export"
    room_out = RESULTS / video / "nerf"

    if not (data_dir / "transforms.json").exists():
        ok = run(
            ["ns-process-data", "images",
             "--data", frames_dir,
             "--output-dir", data_dir,
             "--skip-colmap",
             "--colmap-model-path", sfm_dir / "sparse" / model,
             "--num-downscales", "0"],
            f"{video} / nerf-data", env=nerf_env(), timeout=300,
        )
        if not ok and not (data_dir / "transforms.json").exists():
            print(f"  [{video}] nerf-data failed, skipping NeRF stage")
            return

    config_glob = str(result_dir / video / "nerfacto" / "*" / "config.yml")
    existing_configs = glob.glob(config_glob)
    if not existing_configs:
        run(
            ["ns-train", "nerfacto",
             "--data", data_dir,
             "--output-dir", result_dir,
             "--experiment-name", video,
             "--max-num-iterations", str(NERF_ITERS),
             "--steps-per-save", str(NERF_ITERS),
             "--viewer.quit-on-train-completion", "True",
             "--vis", "viewer",
             "--pipeline.model.implementation", "torch"],
            f"{video} / nerf-train", env=nerf_env(), timeout=900,
        )
        existing_configs = glob.glob(config_glob)

    ply_path = export_dir / "point_cloud.ply"
    if not ply_path.exists() and existing_configs:
        latest_config = max(existing_configs, key=os.path.getmtime)
        run(
            ["ns-export", "pointcloud",
             "--load-config", latest_config,
             "--output-dir", export_dir,
             "--num-points", "500000",
             "--normal-method", "open3d",
             "--save-world-frame", "True"],
            f"{video} / nerf-export", env=nerf_env(), timeout=400,
        )

    room_dims_path = room_out / "nerf_room_dims.json"
    if not room_dims_path.exists() and ply_path.exists():
        run(
            [PY, ROOT / "3d_recons_alternatives" / "nerf_pointcloud_to_room_dims.py",
             "--ply", ply_path,
             "--sfm-dir", sfm_dir,
             "--model", model,
             "--out", room_out],
            f"{video} / nerf-room-dims", timeout=120,
        )


def do_gsplat(video: str, sfm_dir: Path, frames_dir: Path, model: str) -> None:
    data_dir = GSPLAT_DATA / video
    result_dir = GSPLAT_RESULTS / video
    room_out = RESULTS / video / "gsplat"

    if not (data_dir / "sparse").exists():
        ok = run(
            [PY, ROOT / "3d_recons_alternatives" / "prepare_gsplat_dataset.py",
             "--sfm-dir", sfm_dir,
             "--frames-dir", frames_dir,
             "--model", model,
             "--out", data_dir],
            f"{video} / gsplat-prep", timeout=180,
        )
        if not ok and not (data_dir / "sparse").exists():
            print(f"  [{video}] gsplat-prep failed, skipping gsplat stage")
            return

    ckpt_path = result_dir / "ckpts" / f"ckpt_{GSPLAT_STEPS - 1}_rank0.pt"
    if not ckpt_path.exists():
        run(
            [PY, "simple_trainer.py", "default",
             "--data_dir", data_dir,
             "--data_factor", "1",
             "--result_dir", result_dir,
             "--max_steps", str(GSPLAT_STEPS),
             "--disable_viewer"],
            f"{video} / gsplat-train", cwd=GSPLAT_EXAMPLES, env=gsplat_env(), timeout=900,
        )

    ply_path = result_dir / "ply" / "point_cloud.ply"
    if not ply_path.exists() and ckpt_path.exists():
        run(
            [PY, ROOT / "3d_recons_alternatives" / "ckpt_to_ply.py",
             "--ckpt", ckpt_path,
             "--out", ply_path],
            f"{video} / gsplat-ckpt2ply", timeout=120,
        )

    room_dims_path = room_out / "gaussian_room_dims.json"
    if not room_dims_path.exists() and ply_path.exists():
        run(
            [PY, ROOT / "3d_recons_alternatives" / "gaussians_to_room_dims.py",
             "--ply", ply_path,
             "--sfm-dir", sfm_dir,
             "--model", model,
             "--out", room_out],
            f"{video} / gsplat-room-dims", timeout=120,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None)
    parser.add_argument("--skip-nerf", action="store_true")
    parser.add_argument("--skip-gsplat", action="store_true")
    args = parser.parse_args()

    videos = sorted(d.name for d in SFM_ROOT.iterdir() if d.is_dir())
    if args.only:
        videos = [v for v in videos if v == args.only]

    log_path = OUT / "batch_progress.log"
    done, skipped = [], []

    for video in videos:
        if deadline_hit():
            print(f"\n*** DEADLINE ({DEADLINE_HOURS}h) reached — stopping before {video} ***", flush=True)
            skipped.append(video)
            continue

        sfm_dir = SFM_ROOT / video / "sfm"
        frames_dir = SFM_ROOT / video / "frames"
        scale_json = sfm_dir / "scale.json"
        model = best_sparse_model(sfm_dir)

        if model is None or not scale_json.exists():
            print(f"\n--- {video}: no usable sparse model / scale.json, skipping ---", flush=True)
            skipped.append(video)
            continue

        elapsed_h = (time.time() - START_TIME) / 3600
        print(f"\n{'='*70}\n{video}  (model {model})  [elapsed {elapsed_h:.2f}h]\n{'='*70}", flush=True)

        try:
            if not args.skip_nerf:
                do_nerf(video, sfm_dir, frames_dir, model)
        except Exception as e:
            print(f"  [{video}] NeRF stage crashed: {e}", flush=True)

        try:
            if not args.skip_gsplat:
                do_gsplat(video, sfm_dir, frames_dir, model)
        except Exception as e:
            print(f"  [{video}] gsplat stage crashed: {e}", flush=True)

        done.append(video)
        log_path.write_text(json.dumps({"done": done, "skipped": skipped,
                                        "elapsed_hours": round((time.time() - START_TIME) / 3600, 2)},
                                       indent=2))

    total_h = (time.time() - START_TIME) / 3600
    print(f"\n{'='*70}\nBATCH COMPLETE in {total_h:.2f}h — {len(done)} processed, {len(skipped)} skipped\n{'='*70}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
