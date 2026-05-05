"""Loader and validation utilities for human highlight annotations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.video_registry import load_video_registry
from src.utils.time_utils import hhmmss_to_seconds


HUMAN_HIGHLIGHTS_PATH = Path("data/processed/annotations/human_highlights.csv")
REQUIRED_HIGHLIGHT_COLUMNS = [
    "video_id",
    "start_time",
    "end_time",
    "start_seconds",
    "end_seconds",
    "annotator",
    "reason",
    "confidence",
]


@dataclass(frozen=True)
class HighlightAnnotation:
    """One human-labeled highlight interval."""

    video_id: str
    start_time: str
    end_time: str
    start_seconds: float
    end_seconds: float
    annotator: str
    reason: str
    confidence: float


def load_human_highlights(
    path: str | Path = HUMAN_HIGHLIGHTS_PATH,
    video_registry_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load and validate human highlight annotations."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Human highlights file not found: {path}. "
            "Create it from data/processed/annotations/human_highlights_template.csv."
        )

    highlights = pd.read_csv(path)
    registry = None
    if video_registry_path is not None:
        registry = load_video_registry(video_registry_path)
    return validate_human_highlights(highlights, video_registry=registry)


def validate_human_highlights(
    highlights: pd.DataFrame,
    video_registry: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Validate and normalize a human highlight annotation DataFrame."""

    missing_columns = [column for column in REQUIRED_HIGHLIGHT_COLUMNS if column not in highlights]
    if missing_columns:
        raise ValueError(f"Human highlights missing required columns: {missing_columns}")

    normalized = highlights.loc[:, REQUIRED_HIGHLIGHT_COLUMNS].copy()
    _require_non_empty(normalized, ["video_id", "start_time", "end_time", "annotator"])

    normalized["start_seconds"] = pd.to_numeric(normalized["start_seconds"], errors="raise")
    normalized["end_seconds"] = pd.to_numeric(normalized["end_seconds"], errors="raise")
    normalized["confidence"] = pd.to_numeric(normalized["confidence"], errors="raise")
    normalized["reason"] = normalized["reason"].fillna("").astype(str)

    if (normalized["start_seconds"] < 0).any() or (normalized["end_seconds"] < 0).any():
        raise ValueError("start_seconds and end_seconds must be non-negative.")
    if (normalized["end_seconds"] <= normalized["start_seconds"]).any():
        raise ValueError("Every annotation must have end_seconds greater than start_seconds.")
    if ((normalized["confidence"] < 0) | (normalized["confidence"] > 1)).any():
        raise ValueError("confidence must be between 0 and 1.")

    _validate_timestamp_consistency(normalized)
    if video_registry is not None:
        _validate_video_ids(normalized, video_registry)
        _validate_annotation_bounds(normalized, video_registry)

    return normalized


def annotations_to_records(highlights: pd.DataFrame) -> list[HighlightAnnotation]:
    """Convert validated annotations into dataclass records."""

    validated = validate_human_highlights(highlights)
    return [HighlightAnnotation(**_row_to_dict(row)) for _, row in validated.iterrows()]


def _validate_timestamp_consistency(highlights: pd.DataFrame) -> None:
    parsed_start = highlights["start_time"].map(hhmmss_to_seconds)
    parsed_end = highlights["end_time"].map(hhmmss_to_seconds)

    if (parsed_start != highlights["start_seconds"]).any():
        raise ValueError("start_time and start_seconds disagree for at least one annotation.")
    if (parsed_end != highlights["end_seconds"]).any():
        raise ValueError("end_time and end_seconds disagree for at least one annotation.")


def _validate_video_ids(highlights: pd.DataFrame, video_registry: pd.DataFrame) -> None:
    known_video_ids = set(video_registry["video_id"].astype(str))
    unknown_video_ids = sorted(set(highlights["video_id"].astype(str)).difference(known_video_ids))
    if unknown_video_ids:
        raise ValueError(f"Annotations reference unknown video_id values: {unknown_video_ids}")


def _validate_annotation_bounds(highlights: pd.DataFrame, video_registry: pd.DataFrame) -> None:
    durations = video_registry.set_index("video_id")["duration_seconds"].to_dict()
    for _, row in highlights.iterrows():
        duration = durations[row["video_id"]]
        if row["end_seconds"] > duration:
            raise ValueError(
                f"Annotation for video_id={row['video_id']} ends after video duration "
                f"({row['end_seconds']} > {duration})."
            )


def _row_to_dict(row: pd.Series) -> dict[str, Any]:
    return {column: row[column] for column in REQUIRED_HIGHLIGHT_COLUMNS}


def _require_non_empty(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if frame[column].isna().any() or (frame[column].astype(str).str.strip() == "").any():
            raise ValueError(f"{column} must be present and non-empty for every annotation.")
