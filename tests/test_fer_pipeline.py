from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fer2013_loader import (
    FER2013TorchDataset,
    create_synthetic_fer2013,
    load_prepared_split,
)
from src.evaluation.fer_metrics import compute_fer_metrics
from src.visual.emotion_model import build_model


def test_models_build_with_seven_class_output() -> None:
    inputs = torch.zeros(2, 1, 48, 48)

    for model_name in ("baseline", "improved"):
        model = build_model(model_name, num_classes=7)
        outputs = model(inputs)

        assert outputs.shape == (2, 7)


def test_loader_preprocessing_returns_channel_last_arrays() -> None:
    output_dir = PROJECT_ROOT / "outputs" / "test_tmp" / "fer_loader"
    create_synthetic_fer2013(output_dir, random_seed=123, force=True)

    images, labels = load_prepared_split(output_dir, "train")
    assert images.shape[1:] == (48, 48, 1)
    assert labels.dtype == np.int64
    assert images.dtype == np.float32
    assert 0.0 <= float(images.min()) <= float(images.max()) <= 1.0

    dataset = FER2013TorchDataset(images, labels)
    image_tensor, label = dataset[0]

    assert image_tensor.shape == (1, 48, 48)
    assert isinstance(label, int)


def test_fer_metrics_returns_expected_keys() -> None:
    metrics = compute_fer_metrics(
        y_true=[0, 1, 2, 3, 4, 5, 6],
        y_pred=[0, 1, 1, 3, 4, 6, 6],
        class_names=[
            "angry",
            "disgust",
            "fear",
            "happy",
            "sad",
            "surprise",
            "neutral",
        ],
    )

    expected_keys = {
        "accuracy",
        "precision_macro",
        "recall_macro",
        "macro_f1",
        "confusion_matrix",
        "classification_report",
    }

    assert expected_keys.issubset(metrics.keys())
    assert len(metrics["confusion_matrix"]) == 7
