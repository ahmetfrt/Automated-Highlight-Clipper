"""Compatibility wrappers for FER2013 preparation utilities."""

from src.data.fer2013_loader import (
    EMOTION_LABELS,
    FER2013TorchDataset,
    create_synthetic_fer2013,
    create_data_loaders,
    load_metadata,
    load_prepared_split,
    prepare_fer2013,
    validate_fer2013_raw,
)


__all__ = [
    "EMOTION_LABELS",
    "FER2013TorchDataset",
    "create_synthetic_fer2013",
    "create_data_loaders",
    "load_metadata",
    "load_prepared_split",
    "prepare_fer2013",
    "validate_fer2013_raw",
]
