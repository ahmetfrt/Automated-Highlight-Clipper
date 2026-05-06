from __future__ import annotations

import pytest

from src.utils.time_utils import (
    hhmmss_to_seconds,
    seconds_to_hhmmss,
    temporal_iou,
    temporal_overlap,
)


def test_hhmmss_to_seconds() -> None:
    assert hhmmss_to_seconds("00:00:00") == 0
    assert hhmmss_to_seconds("00:01:05") == 65
    assert hhmmss_to_seconds("01:02:03") == 3723
    assert hhmmss_to_seconds("02:03") == 123


def test_seconds_to_hhmmss() -> None:
    assert seconds_to_hhmmss(0) == "00:00:00"
    assert seconds_to_hhmmss(65) == "00:01:05"
    assert seconds_to_hhmmss(3723) == "01:02:03"


def test_invalid_timestamps_raise_errors() -> None:
    with pytest.raises(ValueError):
        hhmmss_to_seconds("00:61:00")
    with pytest.raises(ValueError):
        seconds_to_hhmmss(-1)


def test_temporal_overlap_and_iou() -> None:
    assert temporal_overlap(10, 70, 40, 100) == 30.0
    assert temporal_iou(10, 70, 40, 100) == 30.0 / 90.0
    assert temporal_overlap(0, 10, 10, 20) == 0.0
    assert temporal_iou(0, 10, 10, 20) == 0.0


def test_invalid_interval_raises_error() -> None:
    with pytest.raises(ValueError):
        temporal_overlap(10, 5, 0, 1)
