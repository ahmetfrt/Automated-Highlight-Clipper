"""Compatibility wrappers for FER2013 emotion model definitions."""

from src.visual.emotion_model import (
    BaselineEmotionCNN,
    ImprovedEmotionCNN,
    build_model,
    count_parameters,
    save_checkpoint,
    set_seed,
)


EmotionCNN = ImprovedEmotionCNN


__all__ = [
    "BaselineEmotionCNN",
    "EmotionCNN",
    "ImprovedEmotionCNN",
    "build_model",
    "count_parameters",
    "save_checkpoint",
    "set_seed",
]
