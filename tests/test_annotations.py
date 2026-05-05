from __future__ import annotations

import pandas as pd
import pytest

from src.data.annotations import validate_human_highlights
from src.data.video_registry import validate_video_registry


def _valid_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "video_id": "vod_001",
                "title": "Example VOD",
                "source_url": "https://example.com/vod",
                "platform": "youtube",
                "duration_seconds": 600,
                "genre": "gaming",
                "has_chat": "true",
                "has_visible_face": "false",
                "notes": "Small validation fixture.",
            }
        ]
    )


def _valid_annotations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "video_id": "vod_001",
                "start_time": "00:01:00",
                "end_time": "00:02:00",
                "start_seconds": 60,
                "end_seconds": 120,
                "annotator": "annotator_a",
                "reason": "Exciting moment.",
                "confidence": 0.9,
            }
        ]
    )


def test_video_registry_validation_normalizes_types() -> None:
    registry = validate_video_registry(_valid_registry())

    assert registry.loc[0, "video_id"] == "vod_001"
    assert registry.loc[0, "duration_seconds"] == 600
    assert bool(registry.loc[0, "has_chat"]) is True
    assert bool(registry.loc[0, "has_visible_face"]) is False


def test_annotation_validation_accepts_valid_rows() -> None:
    registry = validate_video_registry(_valid_registry())
    annotations = validate_human_highlights(_valid_annotations(), video_registry=registry)

    assert annotations.loc[0, "start_seconds"] == 60
    assert annotations.loc[0, "end_seconds"] == 120
    assert annotations.loc[0, "confidence"] == 0.9


def test_annotation_validation_rejects_missing_columns() -> None:
    invalid = _valid_annotations().drop(columns=["confidence"])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_human_highlights(invalid)


def test_annotation_validation_rejects_timestamp_mismatch() -> None:
    invalid = _valid_annotations()
    invalid.loc[0, "start_seconds"] = 61

    with pytest.raises(ValueError, match="start_time and start_seconds disagree"):
        validate_human_highlights(invalid)


def test_annotation_validation_rejects_unknown_video_id() -> None:
    registry = validate_video_registry(_valid_registry())
    invalid = _valid_annotations()
    invalid.loc[0, "video_id"] = "missing_vod"

    with pytest.raises(ValueError, match="unknown video_id"):
        validate_human_highlights(invalid, video_registry=registry)
