"""Time conversion and interval-overlap helpers."""

from __future__ import annotations


def hhmmss_to_seconds(timestamp: str) -> int:
    """Convert ``HH:MM:SS`` or ``MM:SS`` into integer seconds."""

    if not isinstance(timestamp, str):
        raise TypeError("timestamp must be a string.")

    parts = timestamp.strip().split(":")
    if len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"Expected HH:MM:SS or MM:SS timestamp, got: {timestamp!r}")

    try:
        hours_int = int(hours)
        minutes_int = int(minutes)
        seconds_int = int(seconds)
    except ValueError as error:
        raise ValueError(f"Timestamp contains non-integer fields: {timestamp!r}") from error

    if hours_int < 0 or minutes_int < 0 or seconds_int < 0:
        raise ValueError(f"Timestamp fields must be non-negative: {timestamp!r}")
    if minutes_int >= 60 or seconds_int >= 60:
        raise ValueError(f"Minutes and seconds must be less than 60: {timestamp!r}")

    return hours_int * 3600 + minutes_int * 60 + seconds_int


def seconds_to_hhmmss(seconds: int | float) -> str:
    """Convert seconds into a zero-padded ``HH:MM:SS`` timestamp."""

    if seconds < 0:
        raise ValueError("seconds must be non-negative.")
    total_seconds = int(round(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    remaining_seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


def temporal_overlap(
    start_a: int | float,
    end_a: int | float,
    start_b: int | float,
    end_b: int | float,
) -> float:
    """Return overlap duration in seconds between two half-open intervals."""

    _validate_interval(start_a, end_a)
    _validate_interval(start_b, end_b)
    return float(max(0.0, min(end_a, end_b) - max(start_a, start_b)))


def temporal_iou(
    start_a: int | float,
    end_a: int | float,
    start_b: int | float,
    end_b: int | float,
) -> float:
    """Return temporal intersection-over-union between two intervals."""

    overlap = temporal_overlap(start_a, end_a, start_b, end_b)
    union = max(end_a, end_b) - min(start_a, start_b)
    if union <= 0:
        return 0.0
    return float(overlap / union)


def _validate_interval(start: int | float, end: int | float) -> None:
    if start < 0 or end < 0:
        raise ValueError("Interval boundaries must be non-negative.")
    if end < start:
        raise ValueError(f"Interval end must be greater than or equal to start: {start}, {end}")
