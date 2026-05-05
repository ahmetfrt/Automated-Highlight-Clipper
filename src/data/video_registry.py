"""Loader and validation utilities for the selected VOD registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


VIDEO_REGISTRY_PATH = Path("data/processed/annotations/video_registry.csv")
REQUIRED_VIDEO_REGISTRY_COLUMNS = [
    "video_id",
    "title",
    "source_url",
    "platform",
    "duration_seconds",
    "genre",
    "has_chat",
    "has_visible_face",
    "notes",
]
BOOLEAN_COLUMNS = ["has_chat", "has_visible_face"]


@dataclass(frozen=True)
class VideoRecord:
    """One selected VOD entry."""

    video_id: str
    title: str
    source_url: str
    platform: str
    duration_seconds: float
    genre: str
    has_chat: bool
    has_visible_face: bool
    notes: str


def load_video_registry(path: str | Path = VIDEO_REGISTRY_PATH) -> pd.DataFrame:
    """Load and validate the selected VOD registry CSV."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Video registry not found: {path}. "
            "Create it from data/processed/annotations/video_registry_template.csv."
        )

    registry = pd.read_csv(path)
    return validate_video_registry(registry)


def validate_video_registry(registry: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a video registry DataFrame."""

    missing_columns = [column for column in REQUIRED_VIDEO_REGISTRY_COLUMNS if column not in registry]
    if missing_columns:
        raise ValueError(f"Video registry missing required columns: {missing_columns}")

    normalized = registry.loc[:, REQUIRED_VIDEO_REGISTRY_COLUMNS].copy()
    _require_non_empty(normalized, ["video_id", "title", "platform"])

    if normalized["video_id"].duplicated().any():
        duplicates = sorted(normalized.loc[normalized["video_id"].duplicated(), "video_id"].unique())
        raise ValueError(f"Video registry contains duplicate video_id values: {duplicates}")

    normalized["duration_seconds"] = pd.to_numeric(
        normalized["duration_seconds"],
        errors="raise",
    )
    if (normalized["duration_seconds"] <= 0).any():
        raise ValueError("duration_seconds must be positive for every video.")

    for column in BOOLEAN_COLUMNS:
        normalized[column] = normalized[column].map(_parse_bool)

    normalized["notes"] = normalized["notes"].fillna("").astype(str)
    return normalized


def video_registry_to_records(registry: pd.DataFrame) -> list[VideoRecord]:
    """Convert a validated registry DataFrame into dataclass records."""

    validated = validate_video_registry(registry)
    return [VideoRecord(**_row_to_dict(row)) for _, row in validated.iterrows()]


def _row_to_dict(row: pd.Series) -> dict[str, Any]:
    return {column: row[column] for column in REQUIRED_VIDEO_REGISTRY_COLUMNS}


def _require_non_empty(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if frame[column].isna().any() or (frame[column].astype(str).str.strip() == "").any():
            raise ValueError(f"{column} must be present and non-empty for every video.")


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value_string = str(value).strip().lower()
    if value_string in {"true", "1", "yes", "y"}:
        return True
    if value_string in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Expected boolean-like value, got: {value!r}")
