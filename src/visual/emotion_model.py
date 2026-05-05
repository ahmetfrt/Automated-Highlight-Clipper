"""CNN architectures for FER2013 facial emotion recognition."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class BaselineEmotionCNN(nn.Module):
    """Small baseline CNN for 48x48 grayscale FER2013 images."""

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features)


class ImprovedEmotionCNN(nn.Module):
    """CNN with batch normalization, dropout, and a compact classifier."""

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(1, 32, dropout=0.10),
            _conv_block(32, 64, dropout=0.15),
            _conv_block(64, 128, dropout=0.20),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.50),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features)


def build_model(model_name: str, num_classes: int = 7) -> nn.Module:
    """Build a FER2013 model by name."""

    normalized_name = model_name.lower().strip()
    if normalized_name in {"baseline", "baseline_cnn"}:
        return BaselineEmotionCNN(num_classes=num_classes)
    if normalized_name in {"improved", "improved_cnn"}:
        return ImprovedEmotionCNN(num_classes=num_classes)
    raise ValueError(f"Unknown FER2013 model: {model_name}")


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    model_name: str,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    history: list[dict[str, float]],
) -> None:
    """Save a model checkpoint with metadata needed for later inference."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "num_classes": metadata["num_classes"],
            "class_names": metadata["class_names"],
            "normalization": metadata["normalization"],
            "metrics": metrics,
            "history": history,
        },
        path,
    )


def set_seed(seed: int) -> None:
    """Set common random seeds for reproducible training runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _conv_block(in_channels: int, out_channels: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout2d(p=dropout),
    )
