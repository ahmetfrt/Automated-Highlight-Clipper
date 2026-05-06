from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.visual.face_detection import detect_largest_face
from src.visual.visual_scoring import (
    build_frame_prediction_row,
    compute_visual_excitement_score,
    aggregate_visual_windows,
)


def test_visual_excitement_score_prioritizes_high_arousal_emotions() -> None:
    surprise_score = compute_visual_excitement_score({"surprise": 1.0})
    happy_score = compute_visual_excitement_score({"happy": 1.0})
    neutral_score = compute_visual_excitement_score({"neutral": 1.0})

    assert surprise_score == 1.0
    assert happy_score > neutral_score
    assert compute_visual_excitement_score({"surprise": 1.0}, face_detected=False) == 0.0


def test_visual_window_aggregation() -> None:
    rows = [
        build_frame_prediction_row(
            video_id="vod_001",
            frame_timestamp_seconds=0,
            frame_path="frame_0.jpg",
            frame_id="frame_0",
            face_detected=True,
            probabilities={"happy": 1.0},
        ),
        build_frame_prediction_row(
            video_id="vod_001",
            frame_timestamp_seconds=30,
            frame_path="frame_30.jpg",
            frame_id="frame_30",
            face_detected=False,
        ),
        build_frame_prediction_row(
            video_id="vod_001",
            frame_timestamp_seconds=61,
            frame_path="frame_61.jpg",
            frame_id="frame_61",
            face_detected=True,
            probabilities={"surprise": 1.0},
        ),
    ]

    windows = aggregate_visual_windows(
        frame_predictions=pd.DataFrame(rows),
        window_seconds=60,
        stride_seconds=60,
        duration_seconds=120,
    )

    assert len(windows) == 2
    assert windows.loc[0, "face_detection_ratio"] == 0.5
    assert windows.loc[0, "dominant_window_emotion"] == "happy"
    assert windows.loc[1, "mean_visual_score"] == 1.0
    assert windows.loc[1, "dominant_window_emotion"] == "surprise"


def test_visual_window_aggregation_handles_empty_frame_predictions() -> None:
    windows = aggregate_visual_windows(
        frame_predictions=pd.DataFrame(),
        window_seconds=60,
        stride_seconds=60,
        duration_seconds=120,
        video_id="vod_empty",
    )

    assert len(windows) == 2
    assert windows["mean_visual_score"].tolist() == [0.0, 0.0]
    assert windows["dominant_window_emotion"].tolist() == ["no_face", "no_face"]


def test_no_face_detection_is_recorded_explicitly() -> None:
    pytest.importorskip("cv2")
    blank_frame = np.zeros((80, 80, 3), dtype=np.uint8)

    detection = detect_largest_face(blank_frame)
    prediction = build_frame_prediction_row(
        video_id="vod_001",
        frame_timestamp_seconds=0,
        frame_path="blank.jpg",
        frame_id="blank",
        face_detected=detection.face_detected,
    )

    assert detection.face_detected is False
    assert detection.bbox_xywh is None
    assert detection.face_crop is None
    assert prediction["dominant_emotion"] == "no_face"
    assert prediction["visual_excitement_score"] == 0.0
