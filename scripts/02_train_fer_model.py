"""Train and compare FER2013 facial emotion recognition models."""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline and improved CNN models on FER2013."
    )
    parser.add_argument(
        "--model",
        choices=["all", "baseline", "improved"],
        default="all",
        help="Which model to train.",
    )
    parser.add_argument(
        "--raw-dir",
        "--dataset-path",
        dest="raw_dir",
        default="data/raw/fer2013",
        help="Raw FER2013 directory.",
    )
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="Prepared FER2013 directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Base output directory for metrics and figures.",
    )
    parser.add_argument("--models-dir", default=None, help="Checkpoint output directory.")
    parser.add_argument(
        "--metrics-dir",
        default=None,
        help="Metric output directory.",
    )
    parser.add_argument(
        "--figures-dir",
        default=None,
        help="Figure output directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size. Defaults to 128, or 14 in smoke-test mode.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Maximum training epochs. Defaults to 30, or 1 in smoke-test mode.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=6,
        help="Early-stopping patience for the improved CNN.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum validation-loss improvement for early stopping.",
    )
    parser.add_argument("--baseline-lr", type=float, default=1e-3, help="Baseline learning rate.")
    parser.add_argument("--improved-lr", type=float, default=7e-4, help="Improved learning rate.")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for the improved CNN.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=466, help="Random seed.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Do not auto-run FER2013 preparation if processed files are missing.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run against a tiny synthetic dataset and write outputs under the output directory.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional cap on training batches per epoch, useful for smoke tests.",
    )
    return parser.parse_args()


def _resolve_run_paths(args: argparse.Namespace) -> argparse.Namespace:
    output_dir = Path(args.output_dir)
    if args.smoke_test and args.output_dir == "outputs":
        output_dir = output_dir / "smoke_test"

    if args.processed_dir is None:
        args.processed_dir = (
            str(output_dir / "processed_fer2013")
            if args.smoke_test
            else "data/processed/fer2013"
        )
    if args.metrics_dir is None:
        args.metrics_dir = str(output_dir / "metrics")
    if args.figures_dir is None:
        args.figures_dir = str(output_dir / "figures")
    if args.models_dir is None:
        args.models_dir = str(
            output_dir / "models" if args.smoke_test else Path("models/checkpoints")
        )
    if args.epochs is None:
        args.epochs = 1 if args.smoke_test else 30
    if args.batch_size is None:
        args.batch_size = 14 if args.smoke_test else 128
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.max_train_batches is not None and args.max_train_batches <= 0:
        raise ValueError("--max-train-batches must be positive when provided.")
    return args


def main() -> None:
    args = parse_args()
    args = _resolve_run_paths(args)
    import pandas as pd

    from src.data.fer2013_loader import create_synthetic_fer2013, prepare_fer2013
    from src.evaluation.fer_metrics import save_comparison_csv
    from src.visual.emotion_model import set_seed

    set_seed(args.seed)
    device = _resolve_device(args.device)

    processed_dir = Path(args.processed_dir)
    if args.smoke_test:
        print("Creating synthetic FER2013 smoke-test data.")
        create_synthetic_fer2013(
            processed_dir=processed_dir,
            random_seed=args.seed,
            force=True,
        )
        if args.max_train_batches is None:
            args.max_train_batches = 1
    elif not _prepared_data_exists(processed_dir):
        if args.skip_prepare:
            raise FileNotFoundError(
                f"Prepared FER2013 files not found in {processed_dir}. "
                "Run scripts/01_prepare_fer2013.py first."
            )
        print("Prepared FER2013 files not found; running preparation first.")
        prepare_fer2013(raw_dir=args.raw_dir, processed_dir=processed_dir, random_seed=args.seed)

    models_to_train = ["baseline", "improved"] if args.model == "all" else [args.model]
    comparison_rows: list[dict[str, Any]] = []

    for model_name in models_to_train:
        row = train_and_evaluate_model(model_name=model_name, args=args, device=device)
        comparison_rows.append(row)

    save_comparison_csv(
        comparison_rows,
        Path(args.metrics_dir) / "fer2013_model_comparison.csv",
    )
    print("Training complete.")
    print(pd.DataFrame(comparison_rows).to_string(index=False))


def train_and_evaluate_model(
    model_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.optim import Adam, AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    from src.data.fer2013_loader import create_data_loaders
    from src.evaluation.fer_metrics import (
        compute_fer_metrics,
        plot_confusion_matrix,
        plot_training_curves,
        save_history_csv,
        save_metrics_json,
    )
    from src.visual.emotion_model import (
        build_model,
        count_parameters,
        save_checkpoint,
    )

    augment_train = model_name == "improved"
    loaders, metadata = create_data_loaders(
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_train=augment_train,
    )

    model = build_model(model_name, num_classes=metadata["num_classes"]).to(device)
    parameter_count = count_parameters(model)
    criterion = nn.CrossEntropyLoss()

    if model_name == "baseline":
        optimizer = Adam(model.parameters(), lr=args.baseline_lr)
        scheduler = None
        use_early_stopping = False
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=args.improved_lr,
            weight_decay=args.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(1, args.patience // 2),
        )
        use_early_stopping = True

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = _train_one_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            max_batches=args.max_train_batches,
        )
        val_loss, val_accuracy, _, _ = _evaluate_model(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            device=device,
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "train_accuracy": float(train_accuracy),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_accuracy),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

        improved = val_loss < best_val_loss - args.min_delta
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if use_early_stopping and epochs_without_improvement >= args.patience:
            print(f"Early stopping {model_name} at epoch {epoch}.")
            break

    training_time = time.perf_counter() - start_time
    model.load_state_dict(best_state)

    test_loss, test_accuracy, y_true, y_pred = _evaluate_model(
        model=model,
        loader=loaders["test"],
        criterion=criterion,
        device=device,
    )
    metrics = compute_fer_metrics(y_true=y_true, y_pred=y_pred, class_names=metadata["class_names"])
    metrics.update(
        {
            "model": model_name,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "parameter_count": int(parameter_count),
            "training_time_seconds": float(training_time),
            "best_epoch": int(best_epoch),
            "early_stopping": use_early_stopping,
            "data_augmentation": augment_train,
            "dataset_source": metadata["source_format"],
            "is_smoke_test": bool(metadata.get("is_smoke_test", False)),
        }
    )

    metrics_dir = Path(args.metrics_dir)
    figures_dir = Path(args.figures_dir)
    models_dir = Path(args.models_dir)
    prefix = f"fer2013_{model_name}_cnn"

    save_metrics_json(metrics, metrics_dir / f"{prefix}_metrics.json")
    save_history_csv(history, metrics_dir / f"{prefix}_history.csv")
    plot_training_curves(
        history=history,
        output_path=figures_dir / f"{prefix}_training_curves.png",
        title=f"FER2013 {model_name.title()} CNN Training Curves",
    )
    plot_confusion_matrix(
        matrix=metrics["confusion_matrix"],
        class_names=metadata["class_names"],
        output_path=figures_dir / f"{prefix}_confusion_matrix.png",
        title=f"FER2013 {model_name.title()} CNN Confusion Matrix",
    )
    save_checkpoint(
        path=models_dir / f"{prefix}.pt",
        model=model,
        model_name=model_name,
        metadata=metadata,
        metrics=metrics,
        history=history,
    )

    return {
        "model": model_name,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision_macro"],
        "recall": metrics["recall_macro"],
        "macro_f1": metrics["macro_f1"],
        "parameter_count": parameter_count,
        "training_time_seconds": training_time,
        "best_epoch": best_epoch,
    }


def _train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_batches: int | None = None,
) -> tuple[float, float]:
    from tqdm import tqdm

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for batch_index, (images, labels) in enumerate(progress, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size
        progress.set_postfix(loss=total_loss / max(total, 1), acc=correct / max(total, 1))

        if max_batches is not None and batch_index >= max_batches:
            break

    return total_loss / total, correct / total


def _evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    import torch

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            predictions = logits.argmax(dim=1)
            total_loss += loss.item() * batch_size
            correct += (predictions == labels).sum().item()
            total += batch_size
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    return total_loss / total, correct / total, y_true, y_pred


def _resolve_device(requested_device: str) -> torch.device:
    import torch

    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def _prepared_data_exists(processed_dir: Path) -> bool:
    expected_files = [processed_dir / f"{split}.npz" for split in ("train", "val", "test")]
    expected_files.append(processed_dir / "metadata.json")
    return all(path.exists() for path in expected_files)


if __name__ == "__main__":
    main()
