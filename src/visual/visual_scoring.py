"""Visual emotion scoring and window aggregation utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


EMOTION_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

# A clear high-arousal weighted average over emotion probabilities.
# Surprise is treated as the strongest highlight signal, happy is also high,
# fear and anger are medium/high arousal, and neutral/sad/disgust are lower
# excitement signals for this first visual-only baseline.
EMOTION_EXCITEMENT_WEIGHTS = {
    "angry": 0.55,
    "disgust": 0.20,
    "fear": 0.70,
    "happy": 0.85,
    "sad": 0.25,
    "surprise": 1.00,
    "neutral": 0.15,
}


def probability_column(emotion: str) -> str:
    """Return the frame-output probability column name for one emotion."""

    return f"prob_{emotion}"


def probability_columns(class_names: Sequence[str] = EMOTION_CLASSES) -> list[str]:
    """Return probability output columns in class-name order."""

    return [probability_column(emotion) for emotion in class_names]


def compute_visual_excitement_score(
    probabilities: Mapping[str, float],
    face_detected: bool = True,
    weights: Mapping[str, float] = EMOTION_EXCITEMENT_WEIGHTS,
) -> float:
    """Compute a frame-level visual excitement score from emotion probabilities."""

    if not face_detected:
        return 0.0

    score = 0.0
    for emotion, weight in weights.items():
        score += float(probabilities.get(emotion, 0.0)) * float(weight)
    return float(max(0.0, min(1.0, score)))


def build_frame_prediction_row(
    video_id: str,
    frame_timestamp_seconds: float,
    frame_path: str,
    frame_id: str,
    face_detected: bool,
    probabilities: Mapping[str, float] | None = None,
    dominant_emotion: str | None = None,
    bbox_xywh: tuple[int, int, int, int] | None = None,
    face_crop_path: str = "",
    class_names: Sequence[str] = EMOTION_CLASSES,
) -> dict[str, Any]:
    """Build one standardized frame-level visual prediction row."""

    probabilities = probabilities or {}
    if dominant_emotion is None:
        dominant_emotion = _dominant_emotion(probabilities) if face_detected else "no_face"

    row: dict[str, Any] = {
        "video_id": video_id,
        "frame_timestamp_seconds": float(frame_timestamp_seconds),
        "frame_id": frame_id,
        "frame_path": frame_path,
        "face_detected": bool(face_detected),
        "dominant_emotion": dominant_emotion,
        "visual_excitement_score": compute_visual_excitement_score(
            probabilities,
            face_detected=face_detected,
        ),
        "face_bbox_x": "",
        "face_bbox_y": "",
        "face_bbox_width": "",
        "face_bbox_height": "",
        "face_crop_path": face_crop_path,
    }
    if bbox_xywh is not None:
        x, y, width, height = bbox_xywh
        row.update(
            {
                "face_bbox_x": int(x),
                "face_bbox_y": int(y),
                "face_bbox_width": int(width),
                "face_bbox_height": int(height),
            }
        )

    for emotion in class_names:
        row[probability_column(emotion)] = float(probabilities.get(emotion, 0.0))
    return row


def aggregate_visual_windows(
    frame_predictions: pd.DataFrame,
    window_seconds: int = 60,
    stride_seconds: int = 30,
    duration_seconds: float | None = None,
    video_id: str | None = None,
    class_names: Sequence[str] = EMOTION_CLASSES,
) -> pd.DataFrame:
    """Aggregate frame-level visual scores into fixed temporal windows."""

    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive.")
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive.")

    frame_predictions = frame_predictions.copy()
    if video_id is None:
        video_id = _infer_video_id(frame_predictions)

    output_columns = [
        "video_id",
        "window_start_seconds",
        "window_end_seconds",
        "mean_visual_score",
        "max_visual_score",
        "face_detection_ratio",
        "dominant_window_emotion",
    ]
    if frame_predictions.empty and duration_seconds is None:
        return pd.DataFrame(columns=output_columns)

    if duration_seconds is None:
        max_timestamp = float(frame_predictions["frame_timestamp_seconds"].max())
        duration_seconds = max(window_seconds, max_timestamp + 1.0)
    if duration_seconds <= 0:
        return pd.DataFrame(columns=output_columns)

    if frame_predictions.empty:
        return _empty_frame_windows(
            video_id=video_id,
            duration_seconds=float(duration_seconds),
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            output_columns=output_columns,
        )

    frame_predictions["face_detected"] = frame_predictions["face_detected"].astype(bool)
    rows: list[dict[str, Any]] = []
    start = 0.0
    while start < float(duration_seconds):
        end = min(start + float(window_seconds), float(duration_seconds))
        mask = (
            (frame_predictions["frame_timestamp_seconds"] >= start)
            & (frame_predictions["frame_timestamp_seconds"] < end)
        )
        window_frame = frame_predictions.loc[mask]
        rows.append(
            {
                "video_id": video_id,
                "window_start_seconds": float(start),
                "window_end_seconds": float(end),
                "mean_visual_score": _mean_or_zero(
                    window_frame,
                    "visual_excitement_score",
                ),
                "max_visual_score": _max_or_zero(
                    window_frame,
                    "visual_excitement_score",
                ),
                "face_detection_ratio": _mean_or_zero(window_frame, "face_detected"),
                "dominant_window_emotion": _dominant_window_emotion(
                    window_frame,
                    class_names=class_names,
                ),
            }
        )
        start += float(stride_seconds)

    return pd.DataFrame(rows, columns=output_columns)


def _dominant_emotion(probabilities: Mapping[str, float]) -> str:
    if not probabilities:
        return "no_face"
    return max(probabilities.items(), key=lambda item: float(item[1]))[0]


def _dominant_window_emotion(
    window_frame: pd.DataFrame,
    class_names: Sequence[str],
) -> str:
    if window_frame.empty or not window_frame["face_detected"].any():
        return "no_face"

    face_rows = window_frame.loc[window_frame["face_detected"]]
    columns = [
        probability_column(emotion)
        for emotion in class_names
        if probability_column(emotion) in face_rows.columns
    ]
    if columns:
        mean_probabilities = face_rows[columns].mean()
        dominant_column = str(mean_probabilities.idxmax())
        return dominant_column.removeprefix("prob_")

    emotion_counts = face_rows["dominant_emotion"].value_counts()
    if emotion_counts.empty:
        return "no_face"
    return str(emotion_counts.idxmax())


def _infer_video_id(frame_predictions: pd.DataFrame) -> str:
    if "video_id" not in frame_predictions or frame_predictions.empty:
        return ""
    video_ids = frame_predictions["video_id"].dropna().astype(str).unique()
    if len(video_ids) == 0:
        return ""
    if len(video_ids) > 1:
        raise ValueError("frame_predictions contains multiple video_id values.")
    return str(video_ids[0])


def _empty_frame_windows(
    video_id: str,
    duration_seconds: float,
    window_seconds: int,
    stride_seconds: int,
    output_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    start = 0.0
    while start < duration_seconds:
        rows.append(
            {
                "video_id": video_id,
                "window_start_seconds": float(start),
                "window_end_seconds": min(start + window_seconds, duration_seconds),
                "mean_visual_score": 0.0,
                "max_visual_score": 0.0,
                "face_detection_ratio": 0.0,
                "dominant_window_emotion": "no_face",
            }
        )
        start += float(stride_seconds)
    return pd.DataFrame(rows, columns=output_columns)


def _mean_or_zero(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    return float(frame[column].astype(float).mean())


def _max_or_zero(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    return float(frame[column].astype(float).max())
