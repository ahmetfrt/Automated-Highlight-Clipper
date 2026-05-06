"""Frame extraction and FER2013 visual emotion inference utilities."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data.fer2013_loader import EMOTION_LABELS, IMAGE_SIZE
from src.visual.emotion_model import build_model
from src.visual.face_detection import detect_largest_face
from src.visual.visual_scoring import (
    build_frame_prediction_row,
)


SELECTED_EMOTION_CHECKPOINT = Path("models/checkpoints/fer2013_improved_cnn.pt")
FRAME_MANIFEST_FILENAME = "frames_manifest.csv"
VIDEO_EXTENSIONS = [".mp4", ".mov", ".mkv", ".avi", ".webm"]


def extract_sampled_frames(
    video_path: str | Path,
    video_id: str,
    output_dir: str | Path = "data/interim/frames",
    sample_rate_fps: float = 1.0,
    overwrite: bool = False,
    max_frames: int | None = None,
) -> pd.DataFrame:
    """Sample frames from a local video and write a frame manifest CSV."""

    cv2 = _import_cv2()
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if sample_rate_fps <= 0:
        raise ValueError("sample_rate_fps must be positive.")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided.")

    frames_dir = Path(output_dir) / video_id
    manifest_path = frames_dir / FRAME_MANIFEST_FILENAME
    if manifest_path.exists() and not overwrite:
        return pd.read_csv(manifest_path)

    frames_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0:
            raise ValueError(f"Unable to read FPS from video file: {video_path}")

        if frame_count > 0:
            rows = _sample_by_timestamp(
                capture=capture,
                video_path=video_path,
                video_id=video_id,
                frames_dir=frames_dir,
                fps=fps,
                frame_count=frame_count,
                sample_rate_fps=sample_rate_fps,
                max_frames=max_frames,
            )
        else:
            rows = _sample_by_frame_index(
                capture=capture,
                video_path=video_path,
                video_id=video_id,
                frames_dir=frames_dir,
                fps=fps,
                sample_rate_fps=sample_rate_fps,
                max_frames=max_frames,
            )
    finally:
        capture.release()

    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False)
    return manifest


def load_frame_manifest(
    video_id: str,
    frames_dir: str | Path = "data/interim/frames",
) -> pd.DataFrame:
    """Load a saved frame manifest for a video."""

    manifest_path = Path(frames_dir) / video_id / FRAME_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Frame manifest not found: {manifest_path}. "
            "Run scripts/03_extract_video_assets.py first or pass --video-path "
            "to scripts/04_run_visual_pipeline.py."
        )
    return pd.read_csv(manifest_path)


def resolve_local_video_path(
    video_id: str,
    explicit_video_path: str | Path | None = None,
    videos_dir: str | Path = "data/raw/videos",
) -> Path:
    """Resolve a local video path from an explicit path or ``video_id``."""

    if explicit_video_path is not None:
        video_path = Path(explicit_video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        return video_path

    videos_dir = Path(videos_dir)
    for extension in VIDEO_EXTENSIONS:
        candidate = videos_dir / f"{video_id}{extension}"
        if candidate.exists():
            return candidate

    supported = ", ".join(VIDEO_EXTENSIONS)
    raise FileNotFoundError(
        f"No local video file found for video_id={video_id!r} in {videos_dir}. "
        f"Use --video-path or place the file as {video_id}<ext> with ext in: "
        f"{supported}."
    )


def load_emotion_model(
    checkpoint_path: str | Path = SELECTED_EMOTION_CHECKPOINT,
    device: str | torch.device = "auto",
) -> tuple[torch.nn.Module, list[str], dict[str, float], torch.device]:
    """Load the selected FER2013 CNN checkpoint for inference."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Selected FER2013 checkpoint not found: {checkpoint_path}. "
            "Train the improved model first or pass --checkpoint-path."
        )

    resolved_device = _resolve_device(device)
    checkpoint = _torch_load_checkpoint(checkpoint_path, resolved_device)
    class_names = list(checkpoint.get("class_names", EMOTION_LABELS))
    model_name = str(checkpoint.get("model_name", "improved"))
    num_classes = int(checkpoint.get("num_classes", len(class_names)))
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    normalization = checkpoint.get("normalization", {"mean": 0.0, "std": 1.0})

    model = build_model(model_name, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(resolved_device)
    model.eval()
    return model, class_names, normalization, resolved_device


def preprocess_face_crop(
    face_crop: np.ndarray,
    normalization: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Convert a face crop into a normalized 1x1x48x48 tensor."""

    cv2 = _import_cv2()
    if face_crop is None or face_crop.size == 0:
        raise ValueError("face_crop must be a non-empty image array.")

    if face_crop.ndim == 2:
        gray = face_crop
    elif face_crop.ndim == 3 and face_crop.shape[2] == 3:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    elif face_crop.ndim == 3 and face_crop.shape[2] == 4:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported face crop shape: {face_crop.shape}")

    resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    image = resized.astype(np.float32)
    if image.max(initial=0.0) > 1.0:
        image = image / 255.0

    normalization = normalization or {"mean": 0.0, "std": 1.0}
    mean = float(normalization.get("mean", 0.0))
    std = float(normalization.get("std", 1.0))
    if std <= 1e-8:
        std = 1.0
    image = (image - mean) / std
    return torch.from_numpy(image).unsqueeze(0).unsqueeze(0)


def predict_face_emotion(
    model: torch.nn.Module,
    face_crop: np.ndarray,
    class_names: list[str],
    normalization: Mapping[str, float],
    device: str | torch.device,
) -> dict[str, float]:
    """Run FER2013 emotion inference on one detected face crop."""

    resolved_device = _resolve_device(device)
    input_tensor = preprocess_face_crop(face_crop, normalization).to(resolved_device)
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return {
        class_name: float(probability)
        for class_name, probability in zip(class_names, probabilities)
    }


def run_visual_inference_on_frames(
    frame_manifest: pd.DataFrame,
    video_id: str,
    checkpoint_path: str | Path = SELECTED_EMOTION_CHECKPOINT,
    faces_dir: str | Path | None = "data/interim/faces",
    save_face_crops: bool = True,
    device: str | torch.device = "auto",
    mock_probabilities: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Detect faces and produce frame-level visual emotion predictions."""

    cv2 = _import_cv2()
    if frame_manifest.empty:
        return pd.DataFrame()

    if mock_probabilities is None:
        model, class_names, normalization, resolved_device = load_emotion_model(
            checkpoint_path=checkpoint_path,
            device=device,
        )
    else:
        model = None
        class_names = list(EMOTION_LABELS)
        normalization = {"mean": 0.0, "std": 1.0}
        resolved_device = _resolve_device("cpu")

    output_faces_dir = None
    if faces_dir is not None and save_face_crops:
        output_faces_dir = Path(faces_dir) / video_id
        output_faces_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, frame_row in frame_manifest.reset_index(drop=True).iterrows():
        frame_path = Path(str(frame_row["frame_path"]))
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise FileNotFoundError(f"Unable to read frame image: {frame_path}")

        frame_id = str(frame_row.get("frame_id", f"frame_{index:06d}"))
        timestamp = float(frame_row["frame_timestamp_seconds"])
        detection = detect_largest_face(frame)
        if not detection.face_detected or detection.face_crop is None:
            rows.append(
                build_frame_prediction_row(
                    video_id=video_id,
                    frame_timestamp_seconds=timestamp,
                    frame_path=str(frame_path),
                    frame_id=frame_id,
                    face_detected=False,
                    class_names=class_names,
                )
            )
            continue

        face_crop_path = ""
        if output_faces_dir is not None:
            face_crop_path = str(output_faces_dir / f"{frame_id}_face.jpg")
            cv2.imwrite(face_crop_path, detection.face_crop)

        if mock_probabilities is not None:
            probabilities = _normalize_mock_probabilities(mock_probabilities)
        else:
            if model is None:
                raise RuntimeError("Model is not loaded for real inference.")
            probabilities = predict_face_emotion(
                model=model,
                face_crop=detection.face_crop,
                class_names=class_names,
                normalization=normalization,
                device=resolved_device,
            )

        rows.append(
            build_frame_prediction_row(
                video_id=video_id,
                frame_timestamp_seconds=timestamp,
                frame_path=str(frame_path),
                frame_id=frame_id,
                face_detected=True,
                probabilities=probabilities,
                bbox_xywh=detection.bbox_xywh,
                face_crop_path=face_crop_path,
                class_names=class_names,
            )
        )

    return pd.DataFrame(rows)


def create_synthetic_smoke_video(
    output_path: str | Path,
    duration_seconds: int = 3,
    fps: int = 5,
) -> Path:
    """Create a tiny synthetic video for smoke-testing frame extraction."""

    cv2 = _import_cv2()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 96, 96
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise ValueError(f"Unable to create synthetic smoke-test video: {output_path}")

    try:
        for frame_index in range(duration_seconds * fps):
            frame = np.full((height, width, 3), 30, dtype=np.uint8)
            center_x = 12 + (frame_index * 5) % (width - 24)
            cv2.circle(frame, (center_x, 48), 12, (180, 180, 180), thickness=-1)
            cv2.putText(
                frame,
                str(frame_index),
                (8, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()

    return output_path


def save_visual_outputs(
    frame_predictions: pd.DataFrame,
    window_scores: pd.DataFrame,
    video_id: str,
    output_dir: str | Path = "data/processed/window_features",
) -> tuple[Path, Path]:
    """Save frame-level predictions and visual window scores as CSV files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_output_path = output_dir / f"{video_id}_visual_frame_predictions.csv"
    window_output_path = output_dir / f"{video_id}_visual_window_scores.csv"
    frame_predictions.to_csv(frame_output_path, index=False)
    window_scores.to_csv(window_output_path, index=False)
    return frame_output_path, window_output_path


def _sample_by_timestamp(
    capture: Any,
    video_path: Path,
    video_id: str,
    frames_dir: Path,
    fps: float,
    frame_count: int,
    sample_rate_fps: float,
    max_frames: int | None,
) -> list[dict[str, Any]]:
    cv2 = _import_cv2()
    rows: list[dict[str, Any]] = []
    duration_seconds = frame_count / fps
    step_seconds = 1.0 / sample_rate_fps
    timestamp = 0.0

    while timestamp < duration_seconds:
        if max_frames is not None and len(rows) >= max_frames:
            break
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        success, frame = capture.read()
        if not success:
            break
        rows.append(
            _write_frame_record(
                frame=frame,
                frames_dir=frames_dir,
                video_path=video_path,
                video_id=video_id,
                frame_index=len(rows),
                timestamp_seconds=timestamp,
            )
        )
        timestamp += step_seconds
    return rows


def _sample_by_frame_index(
    capture: Any,
    video_path: Path,
    video_id: str,
    frames_dir: Path,
    fps: float,
    sample_rate_fps: float,
    max_frames: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_every_n_frames = max(1, int(round(fps / sample_rate_fps)))
    frame_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            break
        if frame_index % sample_every_n_frames == 0:
            if max_frames is not None and len(rows) >= max_frames:
                break
            timestamp = frame_index / fps
            rows.append(
                _write_frame_record(
                    frame=frame,
                    frames_dir=frames_dir,
                    video_path=video_path,
                    video_id=video_id,
                    frame_index=len(rows),
                    timestamp_seconds=timestamp,
                )
            )
        frame_index += 1
    return rows


def _write_frame_record(
    frame: np.ndarray,
    frames_dir: Path,
    video_path: Path,
    video_id: str,
    frame_index: int,
    timestamp_seconds: float,
) -> dict[str, Any]:
    cv2 = _import_cv2()
    frame_id = f"frame_{frame_index:06d}_t{timestamp_seconds:010.2f}"
    frame_path = frames_dir / f"{frame_id}.jpg"
    cv2.imwrite(str(frame_path), frame)
    return {
        "video_id": video_id,
        "frame_id": frame_id,
        "frame_timestamp_seconds": round(float(timestamp_seconds), 3),
        "frame_path": str(frame_path),
        "source_video_path": str(video_path),
    }


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    normalized = str(device).lower().strip()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(normalized)


def _torch_load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return checkpoint


def _normalize_mock_probabilities(
    probabilities: Mapping[str, float],
) -> dict[str, float]:
    values = {emotion: float(probabilities.get(emotion, 0.0)) for emotion in EMOTION_LABELS}
    total = sum(max(0.0, value) for value in values.values())
    if total <= 0:
        return {emotion: 0.0 for emotion in EMOTION_LABELS}
    return {
        emotion: max(0.0, value) / total
        for emotion, value in values.items()
    }


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise ImportError(
            "OpenCV is required for visual frame extraction. "
            "Install project requirements with `pip install -r requirements.txt`."
        ) from error
    return cv2
