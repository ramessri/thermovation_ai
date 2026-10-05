"""
run_mvs.py — dense reconstruction (patch-match stereo + fusion) from an
existing sparse COLMAP model.

Why this shells out to a separate colmap.exe instead of using pycolmap
--------------------------------------------------------------------------
pycolmap.patch_match_stereo() exists in the Python API, but the `pip install
pycolmap` wheel on this platform is a CPU-only build (check with
`python -c "import pycolmap; print(pycolmap.COLMAP_build)"` — it will print
"...without CUDA"), and dense stereo has no CPU fallback: it raises
"Dense stereo reconstruction requires CUDA" outright. This is the same
constraint the sibling `photogram` project hit and documents in its own
CLAUDE.md. The fix is the same one used there: build (or download a
prebuilt) CUDA-enabled `colmap` CLI binary and shell out to it for the
dense-only stages, while everything else in this repo keeps using pycolmap
directly. Undistortion has no CUDA requirement, so it still goes through
pycolmap for consistency with the rest of the codebase.

Set COLMAP_BIN to a CUDA-enabled colmap(.exe). On this machine that's
C:\\thermovation-repos\\colmap-cuda\\bin\\colmap.exe (prebuilt Windows CUDA
release, no source build needed).

Output: <sfm_dir>/dense/fused.ply (plus the COLMAP-format undistorted
workspace under <sfm_dir>/dense/ that patch_match_stereo/stereo_fusion need
— images/, sparse/, stereo/depth_maps, stereo/normal_maps).

Usage:
  python 03_reconstruction/run_mvs.py output/sfm/IMG_3126 dataset/frames/IMG_3126 --model 0
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

COLMAP_BIN = os.environ.get("COLMAP_BIN", "colmap")

# Conservative caps so this fits comfortably in a 10-12 GB card alongside
# whatever else is loaded (matches the values validated in photogram's
# mvs.py on the same class of GPU).
PATCH_MATCH_MAX_IMAGE_SIZE = 1200
PATCH_MATCH_CACHE_SIZE_GB = 8
FUSION_MAX_IMAGE_SIZE = 800


def _require_cuda_colmap() -> None:
    resolved = shutil.which(COLMAP_BIN) or (COLMAP_BIN if Path(COLMAP_BIN).exists() else None)
    if not resolved:
        raise SystemExit(
            f"run_mvs: colmap binary not found at COLMAP_BIN={COLMAP_BIN!r}. "
            f"Set COLMAP_BIN to a CUDA-enabled colmap.exe, e.g.\n"
            f'  $env:COLMAP_BIN = "C:\\thermovation-repos\\colmap-cuda\\bin\\colmap.exe"'
        )
    out = subprocess.run([resolved, "-h"], capture_output=True, text=True, timeout=15)
    if "with CUDA" not in out.stdout:
        raise SystemExit(
            f"run_mvs: colmap binary at {resolved!r} was not built with CUDA "
            f"(patch_match_stereo requires it, no CPU fallback exists). "
            f"First line of `colmap -h`: {out.stdout.splitlines()[0] if out.stdout else '(empty)'}"
        )


def _run_streamed(cmd: list, log_path: Path, progress_re: str | None, on_progress=None) -> None:
    """Run a subprocess, tee output to a log file, and optionally callback on regex matches."""
    last_lines: list[str] = []
    pattern = re.compile(progress_re) if progress_re else None
    with open(log_path, "w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log_fh.write(line)
            last_lines.append(line)
            if len(last_lines) > 50:
                last_lines.pop(0)
            if pattern and on_progress:
                m = pattern.search(line)
                if m:
                    on_progress(m)
        proc.wait()
    if proc.returncode != 0:
        tail = "".join(last_lines[-20:])
        raise RuntimeError(f"{cmd[0]} {cmd[1]} failed (rc={proc.returncode}). Log tail:\n{tail}")


def _count_ply_points(ply_path: Path) -> int:
    with open(ply_path, "rb") as fh:
        for _ in range(50):
            line = fh.readline()
            if not line:
                break
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("element vertex"):
                try:
                    return int(text.split()[-1])
                except (ValueError, IndexError):
                    return -1
            if text == "end_header":
                break
    return -1


def run_mvs(sfm_dir: Path, frames_dir: Path, model: str,
           max_image_size: int = PATCH_MATCH_MAX_IMAGE_SIZE) -> Path:
    import pycolmap

    _require_cuda_colmap()
    resolved_bin = shutil.which(COLMAP_BIN) or COLMAP_BIN

    sparse_in = sfm_dir / "sparse" / model
    if not (sparse_in / "images.bin").exists():
        raise SystemExit(f"run_mvs: no reconstruction at {sparse_in} (expected images.bin)")

    dense_dir = sfm_dir / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)

    print(f"[mvs] undistorting images from {sparse_in} ...")
    t0 = time.perf_counter()
    pycolmap.undistort_images(
        output_path=str(dense_dir),
        input_path=str(sparse_in),
        image_path=str(frames_dir),
    )
    print(f"[mvs] undistortion done in {time.perf_counter() - t0:.0f}s")

    print(f"[mvs] patch_match_stereo (max_image_size={max_image_size}) ...")
    t0 = time.perf_counter()
    pm_cmd = [
        resolved_bin, "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.max_image_size", str(max_image_size),
        "--PatchMatchStereo.geom_consistency", "true",
        "--PatchMatchStereo.cache_size", str(PATCH_MATCH_CACHE_SIZE_GB),
    ]

    def _pm_progress(m):
        done, total = int(m.group(1)), int(m.group(2))
        elapsed = time.perf_counter() - t0
        rate = done / elapsed if elapsed > 0 and done > 0 else 0
        eta = f" (~{int((total - done) / rate / 60)} min left)" if rate > 0 else ""
        print(f"[mvs] patch-match: {done}/{total}{eta}")

    _run_streamed(pm_cmd, dense_dir / "patch_match.log",
                 r"Processing view (\d+) / (\d+)", _pm_progress)
    print(f"[mvs] patch_match_stereo done in {time.perf_counter() - t0:.0f}s")

    fused_ply = dense_dir / "fused.ply"
    fusion_size = min(max_image_size, FUSION_MAX_IMAGE_SIZE)
    print(f"[mvs] stereo_fusion (max_image_size={fusion_size}) ...")
    t0 = time.perf_counter()
    sf_cmd = [
        resolved_bin, "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(fused_ply),
        "--StereoFusion.max_image_size", str(fusion_size),
        "--StereoFusion.min_num_pixels", "5",
    ]

    def _sf_progress(m):
        done, total = int(m.group(1)), int(m.group(2))
        print(f"[mvs] fusion: {done}/{total}")

    _run_streamed(sf_cmd, dense_dir / "stereo_fusion.log",
                 r"Fusing image \[(\d+)/(\d+)\]", _sf_progress)
    print(f"[mvs] stereo_fusion done in {time.perf_counter() - t0:.0f}s")

    # Depth/normal maps are large (can be several GB for long videos) and
    # aren't needed once fusion has produced the point cloud.
    for subdir in ("depth_maps", "normal_maps", "consistency_graphs"):
        d = dense_dir / "stereo" / subdir
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

    n_pts = _count_ply_points(fused_ply)
    print(f"[mvs] fused point cloud: {n_pts:,} points -> {fused_ply}")
    return fused_ply


def main():
    parser = argparse.ArgumentParser(description="Dense reconstruction (CUDA patch-match stereo)")
    parser.add_argument("sfm_dir", type=Path, help="SfM output dir (contains sparse/<model>)")
    parser.add_argument("frames_dir", type=Path, help="Frame images dir used for SfM")
    parser.add_argument("--model", default="0", help="Sparse model index (default 0)")
    parser.add_argument("--max-image-size", type=int, default=PATCH_MATCH_MAX_IMAGE_SIZE,
                        help=f"Downsize images before patch-match to fit VRAM "
                             f"(default {PATCH_MATCH_MAX_IMAGE_SIZE})")
    args = parser.parse_args()

    try:
        run_mvs(args.sfm_dir, args.frames_dir, args.model, args.max_image_size)
    except RuntimeError as e:
        print(f"FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
