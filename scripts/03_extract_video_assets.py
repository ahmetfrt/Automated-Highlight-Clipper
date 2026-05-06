"""Extract sampled visual frames from a local source video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample frames from a registered local video file.",
    )
    parser.add_argument("--video-id", help="Video ID from the VOD registry.")
    parser.add_argument("--video-path", help="Path to the local video file.")
    parser.add_argument(
        "--videos-dir",
        default="data/raw/videos",
        help="Directory used to resolve <video_id>.mp4 when --video-path is omitted.",
    )
    parser.add_argument(
        "--registry-path",
        default="data/processed/annotations/video_registry.csv",
        help="Optional VOD registry CSV used to validate --video-id.",
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Project YAML config with video.frame_sample_rate.",
    )
    parser.add_argument(
        "--frame-sample-rate",
        type=float,
        default=None,
        help="Frame sample rate in FPS. Defaults to config/default.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/interim/frames",
        help="Base directory for extracted frame folders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite the frame manifest and sampled frame files.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for quick local checks.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Create a tiny synthetic video and sample a few frames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.visual.visual_features import (
        create_synthetic_smoke_video,
        extract_sampled_frames,
        resolve_local_video_path,
    )

    sample_rate = args.frame_sample_rate
    if sample_rate is None:
        sample_rate = _load_frame_sample_rate(args.config)

    if args.smoke_test:
        video_id = args.video_id or "visual_smoke_test"
        video_path = create_synthetic_smoke_video(
            Path("data/interim/smoke_tests") / f"{video_id}.mp4",
        )
        max_frames = args.max_frames or 3
        overwrite = True
    else:
        if not args.video_id:
            raise ValueError("--video-id is required unless --smoke-test is used.")
        video_id = args.video_id
        _validate_registered_video(video_id, args.registry_path)
        video_path = resolve_local_video_path(
            video_id=video_id,
            explicit_video_path=args.video_path,
            videos_dir=args.videos_dir,
        )
        max_frames = args.max_frames
        overwrite = args.overwrite

    manifest = extract_sampled_frames(
        video_path=video_path,
        video_id=video_id,
        output_dir=args.output_dir,
        sample_rate_fps=sample_rate,
        overwrite=overwrite,
        max_frames=max_frames,
    )

    summary = {
        "video_id": video_id,
        "video_path": str(video_path),
        "sample_rate_fps": sample_rate,
        "num_frames": int(len(manifest)),
        "frame_manifest_path": str(
            Path(args.output_dir) / video_id / "frames_manifest.csv"
        ),
    }
    print(json.dumps(summary, indent=2))


def _load_frame_sample_rate(config_path: str | Path) -> float:
    path = Path(config_path)
    if not path.exists():
        return 1.0
    with path.open("r", encoding="utf-8") as file:
        config: dict[str, Any] = yaml.safe_load(file) or {}
    return float(config.get("video", {}).get("frame_sample_rate", 1.0))


def _validate_registered_video(video_id: str, registry_path: str | Path) -> None:
    from src.data.video_registry import load_video_registry

    path = Path(registry_path)
    if not path.exists():
        print(f"Warning: registry file not found; skipping registry check: {path}")
        return

    registry = load_video_registry(path)
    video_ids = set(registry["video_id"].astype(str))
    if video_id not in video_ids:
        raise ValueError(f"video_id={video_id!r} is not present in {path}.")


if __name__ == "__main__":
    main()
