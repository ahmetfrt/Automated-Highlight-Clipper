"""Metrics and figures for FER2013 emotion classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_fer_metrics(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
) -> dict[str, Any]:
    """Compute classification metrics for FER2013 predictions."""

    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }


def save_metrics_json(metrics: dict[str, Any], output_path: str | Path) -> None:
    """Save metrics as pretty JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def save_history_csv(history: list[dict[str, float]], output_path: str | Path) -> None:
    """Save epoch-level training history."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_path, index=False)


def save_comparison_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    """Save model-comparison metrics."""

    columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "macro_f1",
        "parameter_count",
        "training_time_seconds",
        "best_epoch",
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)


def plot_training_curves(
    history: list[dict[str, float]],
    output_path: str | Path,
    title: str,
) -> None:
    """Plot loss and accuracy curves for one training run."""

    frame = pd.DataFrame(history)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="Train")
    axes[0].plot(frame["epoch"], frame["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(frame["epoch"], frame["train_accuracy"], label="Train")
    axes[1].plot(frame["epoch"], frame["val_accuracy"], label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(
    matrix: list[list[int]],
    class_names: list[str],
    output_path: str | Path,
    title: str,
) -> None:
    """Plot a confusion matrix heatmap."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
