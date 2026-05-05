"""Compatibility wrappers for timestamp and interval helper utilities."""

from src.utils.time_utils import hhmmss_to_seconds, seconds_to_hhmmss, temporal_iou, temporal_overlap


def seconds_to_timestamp(seconds: int | float) -> str:
    """Convert seconds into a zero-padded ``HH:MM:SS`` timestamp."""

    return seconds_to_hhmmss(seconds)


__all__ = [
    "hhmmss_to_seconds",
    "seconds_to_hhmmss",
    "seconds_to_timestamp",
    "temporal_iou",
    "temporal_overlap",
]
