"""Run the FER2013 visual emotion pipeline on sampled video frames."""

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
        description="Sample frames, detect faces, and score visual emotion windows.",
    )
    parser.add_argument("--video-id", help="Video ID from the VOD registry.")
    parser.add_argument("--video-path", help="Optional path to the local video file.")
    parser.add_argument(
        "--videos-dir",
        default="data/raw/videos",
        help="Directory used to resolve <video_id>.mp4 when --video-path is omitted.",
    )
    parser.add_argument(
        "--registry-path",
        default="data/processed/annotations/video_registry.csv",
        help="Optional VOD registry CSV used for video_id and duration validation.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="models/checkpoints/fer2013_improved_cnn.pt",
        help="Selected FER2013 improved CNN checkpoint.",
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Project YAML config with frame and window settings.",
    )
    parser.add_argument(
        "--frame-sample-rate",
        type=float,
        default=None,
        help="Frame sample rate in FPS. Defaults to config/default.yaml.",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=None,
        help="Window length in seconds. Defaults to config/default.yaml.",
    )
    parser.add_argument(
        "--window-stride-seconds",
        type=int,
        default=None,
        help="Window stride in seconds. Defaults to config/default.yaml.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Optional duration override for window aggregation.",
    )
    parser.add_argument(
        "--frames-dir",
        default="data/interim/frames",
        help="Base directory for sampled frame folders.",
    )
    parser.add_argument(
        "--faces-dir",
        default="data/interim/faces",
        help="Base directory for optional detected face crops.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/window_features",
        help="Directory for frame predictions and visual window scores.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, etc.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame cap for quick local checks.",
    )
    parser.add_argument(
        "--overwrite-frames",
        action="store_true",
        help="Resample frames even if a frame manifest already exists.",
    )
    parser.add_argument(
        "--no-save-face-crops",
        action="store_true",
        help="Do not save detected face crops under data/interim/faces/.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run on a tiny synthetic video with mock probabilities if a face appears.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.visual.visual_features import (
        create_synthetic_smoke_video,
        extract_sampled_frames,
        load_frame_manifest,
        resolve_local_video_path,
        run_visual_inference_on_frames,
        save_visual_outputs,
    )
    from src.visual.visual_scoring import aggregate_visual_windows

    defaults = _load_defaults(args.config)
    sample_rate = args.frame_sample_rate or defaults["frame_sample_rate"]
    window_seconds = args.window_seconds or defaults["window_seconds"]
    stride_seconds = args.window_stride_seconds or defaults["window_stride_seconds"]

    if args.smoke_test:
        video_id = args.video_id or "visual_smoke_test"
        video_path = create_synthetic_smoke_video(
            Path("data/interim/smoke_tests") / f"{video_id}.mp4",
        )
        frame_manifest = extract_sampled_frames(
            video_path=video_path,
            video_id=video_id,
            output_dir=args.frames_dir,
            sample_rate_fps=sample_rate,
            overwrite=True,
            max_frames=args.max_frames or 3,
        )
        duration_seconds = args.duration_seconds or 3.0
        mock_probabilities = {"surprise": 0.55, "happy": 0.35, "fear": 0.10}
    else:
        if not args.video_id:
            raise ValueError("--video-id is required unless --smoke-test is used.")
        video_id = args.video_id
        duration_seconds = args.duration_seconds
        registry_duration = _validate_and_get_duration(video_id, args.registry_path)
        if duration_seconds is None:
            duration_seconds = registry_duration

        if args.video_path is not None:
            video_path = resolve_local_video_path(
                video_id=video_id,
                explicit_video_path=args.video_path,
                videos_dir=args.videos_dir,
            )
            frame_manifest = extract_sampled_frames(
                video_path=video_path,
                video_id=video_id,
                output_dir=args.frames_dir,
                sample_rate_fps=sample_rate,
                overwrite=args.overwrite_frames,
                max_frames=args.max_frames,
            )
        else:
            try:
                frame_manifest = load_frame_manifest(video_id, args.frames_dir)
            except FileNotFoundError:
                video_path = resolve_local_video_path(
                    video_id=video_id,
                    videos_dir=args.videos_dir,
                )
                frame_manifest = extract_sampled_frames(
                    video_path=video_path,
                    video_id=video_id,
                    output_dir=args.frames_dir,
                    sample_rate_fps=sample_rate,
                    overwrite=args.overwrite_frames,
                    max_frames=args.max_frames,
                )
        mock_probabilities = None

    frame_predictions = run_visual_inference_on_frames(
        frame_manifest=frame_manifest,
        video_id=video_id,
        checkpoint_path=args.checkpoint_path,
        faces_dir=args.faces_dir,
        save_face_crops=not args.no_save_face_crops,
        device=args.device,
        mock_probabilities=mock_probabilities,
    )
    window_scores = aggregate_visual_windows(
        frame_predictions=frame_predictions,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        duration_seconds=duration_seconds,
        video_id=video_id,
    )
    frame_output_path, window_output_path = save_visual_outputs(
        frame_predictions=frame_predictions,
        window_scores=window_scores,
        video_id=video_id,
        output_dir=args.output_dir,
    )

    summary = {
        "video_id": video_id,
        "num_frames": int(len(frame_predictions)),
        "num_windows": int(len(window_scores)),
        "frame_predictions_path": str(frame_output_path),
        "visual_window_scores_path": str(window_output_path),
        "face_detection_ratio": _overall_face_detection_ratio(frame_predictions),
    }
    print(json.dumps(summary, indent=2))


def _load_defaults(config_path: str | Path) -> dict[str, Any]:
    defaults = {
        "frame_sample_rate": 1.0,
        "window_seconds": 60,
        "window_stride_seconds": 30,
    }
    path = Path(config_path)
    if not path.exists():
        return defaults
    with path.open("r", encoding="utf-8") as file:
        config: dict[str, Any] = yaml.safe_load(file) or {}
    defaults["frame_sample_rate"] = float(
        config.get("video", {}).get("frame_sample_rate", defaults["frame_sample_rate"])
    )
    defaults["window_seconds"] = int(
        config.get("windows", {}).get("window_seconds", defaults["window_seconds"])
    )
    defaults["window_stride_seconds"] = int(
        config.get("windows", {}).get(
            "window_stride_seconds",
            defaults["window_stride_seconds"],
        )
    )
    return defaults


def _validate_and_get_duration(
    video_id: str,
    registry_path: str | Path,
) -> float | None:
    from src.data.video_registry import load_video_registry

    path = Path(registry_path)
    if not path.exists():
        print(f"Warning: registry file not found; skipping registry check: {path}")
        return None

    registry = load_video_registry(path)
    row = registry.loc[registry["video_id"].astype(str) == video_id]
    if row.empty:
        raise ValueError(f"video_id={video_id!r} is not present in {path}.")
    return float(row.iloc[0]["duration_seconds"])


def _overall_face_detection_ratio(frame_predictions: Any) -> float:
    if frame_predictions.empty or "face_detected" not in frame_predictions:
        return 0.0
    return float(frame_predictions["face_detected"].astype(bool).mean())


if __name__ == "__main__":
    main()
