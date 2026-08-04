"""Extract frames from a video at a fixed sampling rate for dataset building."""

import argparse
from pathlib import Path

import cv2


def extract_frames(video_path: Path, out_dir: Path, fps: float,
                   max_dim: int | None = None) -> int:
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(int(round(src_fps / fps)), 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    idx = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            if max_dim and max(frame.shape[:2]) > max_dim:
                s = max_dim / max(frame.shape[:2])
                frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / f"{stem}_{saved:04d}.jpg"), frame)
            saved += 1
        idx += 1
    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video")
    parser.add_argument("video", type=Path, help="Path to input video")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory (default: dataset/frames/<video_stem>)")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Sampling rate in frames per second (default: 1.0)")
    parser.add_argument("--max-dim", type=int, default=None,
                        help="Downscale so the longest side is at most this many pixels")
    args = parser.parse_args()

    out_dir = args.out or Path("dataset/frames") / args.video.stem
    saved = extract_frames(args.video, out_dir, args.fps, args.max_dim)
    print(f"Saved {saved} frames to {out_dir}")


if __name__ == "__main__":
    main()
