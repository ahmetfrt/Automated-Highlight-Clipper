"""Lightweight face detection helpers for sampled video frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CASCADE_FILENAME = "haarcascade_frontalface_default.xml"


@dataclass(frozen=True)
class FaceDetectionResult:
    """Result for the largest detected face in one frame."""

    face_detected: bool
    bbox_xywh: tuple[int, int, int, int] | None
    face_crop: np.ndarray | None
    detector_name: str = "opencv_haar"


def detect_largest_face(
    frame: np.ndarray,
    cascade_path: str | Path | None = None,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: tuple[int, int] = (24, 24),
    padding_ratio: float = 0.15,
) -> FaceDetectionResult:
    """Detect the largest frontal face in a BGR or grayscale frame.

    The project uses OpenCV's Haar cascade because it is lightweight, local, and
    already listed in the project requirements. If no face is found, the caller
    receives an explicit ``face_detected=False`` result instead of an exception.
    """

    cv2 = _import_cv2()
    if frame is None or frame.size == 0:
        raise ValueError("frame must be a non-empty image array.")

    gray_frame = _to_grayscale(frame)
    cascade = _load_haar_cascade(cv2, cascade_path)
    detections = cascade.detectMultiScale(
        gray_frame,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )

    if len(detections) == 0:
        return FaceDetectionResult(
            face_detected=False,
            bbox_xywh=None,
            face_crop=None,
        )

    x, y, width, height = max(detections, key=lambda box: int(box[2]) * int(box[3]))
    padded_bbox = _pad_bbox(
        x=int(x),
        y=int(y),
        width=int(width),
        height=int(height),
        image_width=int(frame.shape[1]),
        image_height=int(frame.shape[0]),
        padding_ratio=padding_ratio,
    )
    crop_x, crop_y, crop_width, crop_height = padded_bbox
    face_crop = frame[
        crop_y : crop_y + crop_height,
        crop_x : crop_x + crop_width,
    ].copy()

    return FaceDetectionResult(
        face_detected=True,
        bbox_xywh=padded_bbox,
        face_crop=face_crop,
    )


def _to_grayscale(frame: np.ndarray) -> np.ndarray:
    cv2 = _import_cv2()
    if frame.ndim == 2:
        return frame
    if frame.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image array, got shape {frame.shape}.")
    if frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Expected 1, 3, or 4 channels, got shape {frame.shape}.")


def _load_haar_cascade(cv2: Any, cascade_path: str | Path | None) -> Any:
    if cascade_path is None:
        cascade_path = Path(cv2.data.haarcascades) / DEFAULT_CASCADE_FILENAME
    else:
        cascade_path = Path(cascade_path)

    if not cascade_path.exists():
        raise FileNotFoundError(f"OpenCV Haar cascade not found: {cascade_path}")

    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise ValueError(f"Unable to load OpenCV Haar cascade: {cascade_path}")
    return cascade


def _pad_bbox(
    x: int,
    y: int,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    pad_x = int(round(width * padding_ratio))
    pad_y = int(round(height * padding_ratio))
    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(image_width, x + width + pad_x)
    bottom = min(image_height, y + height + pad_y)
    return left, top, right - left, bottom - top


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise ImportError(
            "OpenCV is required for visual frame extraction and face detection. "
            "Install project requirements with `pip install -r requirements.txt`."
        ) from error
    return cv2
