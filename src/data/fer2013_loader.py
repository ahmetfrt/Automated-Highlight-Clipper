"""FER2013 loading, preprocessing, and PyTorch dataset utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset


IMAGE_SIZE = 48
EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

CLASS_NAME_ALIASES = {
    "0": 0,
    "angry": 0,
    "anger": 0,
    "1": 1,
    "disgust": 1,
    "2": 2,
    "fear": 2,
    "3": 3,
    "happy": 3,
    "happiness": 3,
    "4": 4,
    "sad": 4,
    "sadness": 4,
    "5": 5,
    "surprise": 5,
    "surprised": 5,
    "6": 6,
    "neutral": 6,
}


class FER2013TorchDataset(Dataset):
    """PyTorch dataset for normalized FER2013 arrays."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: Any | None = None,
    ) -> None:
        self.images = images.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_array = self.images[index]
        if image_array.ndim == 2:
            image = torch.from_numpy(image_array).unsqueeze(0)
        elif image_array.ndim == 3 and image_array.shape[-1] == 1:
            image = torch.from_numpy(image_array).permute(2, 0, 1)
        elif image_array.ndim == 3 and image_array.shape[0] == 1:
            image = torch.from_numpy(image_array)
        else:
            raise ValueError(
                "Expected FER2013 image shape 48x48, 48x48x1, or 1x48x48; "
                f"got {image_array.shape}."
            )
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[index])


def prepare_fer2013(
    raw_dir: str | Path = "data/raw/fer2013",
    processed_dir: str | Path = "data/processed/fer2013",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 466,
    force: bool = False,
) -> dict[str, Any]:
    """Prepare FER2013 into normalized train/val/test ``.npz`` files.

    Supported raw layouts:
    - Kaggle-style ``fer2013.csv`` with ``emotion``, ``pixels``, and optional
      ``Usage`` columns.
    - Image folders such as ``train/<class>``, ``val/<class>``, ``test/<class>``
      or a single ``<class>`` folder layout that needs splitting.
    """

    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    if _prepared_files_exist(processed_dir) and not force:
        return load_metadata(processed_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"FER2013 raw directory not found: {raw_dir}. "
            "Place the dataset under data/raw/fer2013/."
        )

    csv_path = _find_csv(raw_dir)
    if csv_path is not None:
        images, labels, usage = _load_fer2013_csv(csv_path)
        class_names = EMOTION_LABELS
        splits = _split_arrays(
            images=images,
            labels=labels,
            usage=usage,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed,
        )
        source_format = "csv"
    else:
        splits, class_names = _load_image_folder_splits(
            raw_dir=raw_dir,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed,
        )
        source_format = "image_folders"

    if not splits:
        raise ValueError(f"No FER2013 examples found under {raw_dir}.")

    return _write_prepared_dataset(
        splits=splits,
        processed_dir=processed_dir,
        class_names=class_names,
        source_format=source_format,
        raw_dir=str(raw_dir),
        random_seed=random_seed,
    )


def create_synthetic_fer2013(
    processed_dir: str | Path,
    train_per_class: int = 2,
    val_per_class: int = 1,
    test_per_class: int = 1,
    random_seed: int = 466,
    force: bool = True,
) -> dict[str, Any]:
    """Create a tiny synthetic FER2013-like dataset for smoke tests.

    The generated data is only for validating code paths. It is not a real
    benchmark and must not be reported as FER2013 performance.
    """

    processed_dir = Path(processed_dir)
    if _prepared_files_exist(processed_dir) and not force:
        return load_metadata(processed_dir)

    rng = np.random.default_rng(random_seed)
    split_sizes = {
        "train": train_per_class,
        "val": val_per_class,
        "test": test_per_class,
    }
    splits: dict[str, dict[str, np.ndarray]] = {}

    for split_name, examples_per_class in split_sizes.items():
        images: list[np.ndarray] = []
        labels: list[int] = []
        for label in range(len(EMOTION_LABELS)):
            for example_index in range(examples_per_class):
                base_level = (label + 1) / (len(EMOTION_LABELS) + 1)
                noise = rng.normal(loc=0.0, scale=0.04, size=(IMAGE_SIZE, IMAGE_SIZE, 1))
                image = np.clip(base_level + noise, 0.0, 1.0).astype(np.float32)
                image[label : IMAGE_SIZE : len(EMOTION_LABELS), :, 0] *= 0.75
                image[:, example_index % IMAGE_SIZE, 0] = base_level
                images.append(image)
                labels.append(label)

        splits[split_name] = {
            "images": np.stack(images).astype(np.float32),
            "labels": np.asarray(labels, dtype=np.int64),
        }

    return _write_prepared_dataset(
        splits=splits,
        processed_dir=processed_dir,
        class_names=EMOTION_LABELS,
        source_format="synthetic_smoke_test",
        raw_dir="synthetic",
        random_seed=random_seed,
        extra_metadata={"is_smoke_test": True},
    )


def validate_fer2013_raw(raw_dir: str | Path = "data/raw/fer2013") -> dict[str, Any]:
    """Validate that a raw FER2013 directory has a supported layout."""

    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"FER2013 raw directory not found: {raw_dir}")

    csv_path = _find_csv(raw_dir)
    if csv_path is not None:
        frame = pd.read_csv(csv_path)
        required_columns = {"emotion", "pixels"}
        missing_columns = required_columns.difference(frame.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{csv_path} is missing required columns: {missing}")
        labels = frame["emotion"].astype(np.int64).to_numpy()
        return {
            "source_format": "csv",
            "path": str(csv_path),
            "num_examples": int(len(frame)),
            "columns": list(frame.columns),
            "class_counts": _class_counts(labels, EMOTION_LABELS),
        }

    split_dirs = _discover_split_dirs(raw_dir)
    if split_dirs:
        summary = {
            split_name: _count_images_by_class(split_dir)
            for split_name, split_dir in split_dirs.items()
        }
        if "train" not in summary or "test" not in summary:
            raise ValueError(
                "Image-folder FER2013 layout should include at least train/ "
                "and test/ split directories."
            )
        return {"source_format": "image_folders", "path": str(raw_dir), "splits": summary}

    class_counts = _count_images_by_class(raw_dir)
    if class_counts:
        return {
            "source_format": "image_folders_unsplit",
            "path": str(raw_dir),
            "class_counts": class_counts,
        }

    raise ValueError(
        "No supported FER2013 layout found. Expected fer2013.csv or "
        "train/<class_name>/ and test/<class_name>/ image folders."
    )


def _write_prepared_dataset(
    splits: dict[str, dict[str, np.ndarray]],
    processed_dir: Path,
    class_names: list[str],
    source_format: str,
    raw_dir: str,
    random_seed: int,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_images = _ensure_channel_last(splits["train"]["images"])
    mean = float(train_images.mean())
    std = float(train_images.std())
    if std <= 1e-8:
        std = 1.0

    processed_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_data in splits.items():
        np.savez_compressed(
            processed_dir / f"{split_name}.npz",
            images=_ensure_channel_last(split_data["images"]).astype(np.float32),
            labels=split_data["labels"].astype(np.int64),
        )

    metadata = {
        "source_format": source_format,
        "raw_dir": raw_dir,
        "processed_dir": str(processed_dir),
        "image_size": IMAGE_SIZE,
        "pixel_range": [0.0, 1.0],
        "normalization": {"mean": mean, "std": std},
        "class_names": class_names,
        "num_classes": len(class_names),
        "random_seed": random_seed,
        "split_counts": {
            split_name: _class_counts(split_data["labels"], class_names)
            for split_name, split_data in splits.items()
        },
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    metadata_path = processed_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return metadata


def load_prepared_split(
    processed_dir: str | Path,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one prepared FER2013 split."""

    split_path = Path(processed_dir) / f"{split}.npz"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Prepared split not found: {split_path}. "
            "Run scripts/01_prepare_fer2013.py first."
        )

    with np.load(split_path) as data:
        images = _ensure_channel_last(data["images"].astype(np.float32))
        labels = data["labels"].astype(np.int64)
        return images, labels


def load_metadata(processed_dir: str | Path) -> dict[str, Any]:
    """Load prepared FER2013 metadata."""

    metadata_path = Path(processed_dir) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"FER2013 metadata not found: {metadata_path}. "
            "Run scripts/01_prepare_fer2013.py first."
        )
    with metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def create_data_loaders(
    processed_dir: str | Path = "data/processed/fer2013",
    batch_size: int = 128,
    num_workers: int = 0,
    augment_train: bool = False,
) -> tuple[dict[str, DataLoader], dict[str, Any]]:
    """Create train, validation, and test loaders from prepared arrays."""

    from torchvision import transforms

    metadata = load_metadata(processed_dir)
    mean = metadata["normalization"]["mean"]
    std = metadata["normalization"]["std"]

    train_transform_steps: list[Any] = []
    if augment_train:
        train_transform_steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10, fill=0.0),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.08, 0.08),
                    scale=(0.95, 1.05),
                    fill=0.0,
                ),
            ]
        )
    train_transform_steps.append(transforms.Normalize(mean=[mean], std=[std]))

    train_transform = transforms.Compose(train_transform_steps)
    eval_transform = transforms.Normalize(mean=[mean], std=[std])

    loaders: dict[str, DataLoader] = {}
    for split_name in ("train", "val", "test"):
        images, labels = load_prepared_split(processed_dir, split_name)
        dataset = FER2013TorchDataset(
            images=images,
            labels=labels,
            transform=train_transform if split_name == "train" else eval_transform,
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split_name == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return loaders, metadata


def _prepared_files_exist(processed_dir: Path) -> bool:
    expected_files = [processed_dir / f"{split}.npz" for split in ("train", "val", "test")]
    expected_files.append(processed_dir / "metadata.json")
    return all(path.exists() for path in expected_files)


def _find_csv(raw_dir: Path) -> Path | None:
    preferred = raw_dir / "fer2013.csv"
    if preferred.exists():
        return preferred

    csv_files = sorted(raw_dir.glob("*.csv"))
    return csv_files[0] if csv_files else None


def _load_fer2013_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, pd.Series | None]:
    frame = pd.read_csv(csv_path)
    required_columns = {"emotion", "pixels"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    images = np.stack(frame["pixels"].map(_parse_pixel_string).to_numpy())
    labels = frame["emotion"].astype(np.int64).to_numpy()
    usage = frame["Usage"] if "Usage" in frame.columns else None
    return images, labels, usage


def _parse_pixel_string(pixel_string: str) -> np.ndarray:
    pixels = np.fromstring(pixel_string, sep=" ", dtype=np.float32)
    expected_pixels = IMAGE_SIZE * IMAGE_SIZE
    if pixels.size != expected_pixels:
        raise ValueError(
            f"Expected {expected_pixels} pixels for a FER2013 image, got {pixels.size}."
        )
    return (pixels.reshape(IMAGE_SIZE, IMAGE_SIZE) / 255.0).astype(np.float32)


def _split_arrays(
    images: np.ndarray,
    labels: np.ndarray,
    usage: pd.Series | None,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    if usage is not None:
        usage_normalized = usage.astype(str).str.lower().str.replace(" ", "", regex=False)
        train_mask = usage_normalized.isin({"training", "train"})
        val_mask = usage_normalized.isin({"publictest", "validation", "valid", "val"})
        test_mask = usage_normalized.isin({"privatetest", "test"})

        if train_mask.any() and test_mask.any():
            train_images = images[train_mask.to_numpy()]
            train_labels = labels[train_mask.to_numpy()]
            if val_mask.any():
                val_images = images[val_mask.to_numpy()]
                val_labels = labels[val_mask.to_numpy()]
            else:
                train_images, val_images, train_labels, val_labels = _safe_train_test_split(
                    train_images,
                    train_labels,
                    test_size=val_ratio,
                    random_state=random_seed,
                    stratify=train_labels,
                )
            return {
                "train": {"images": train_images, "labels": train_labels},
                "val": {"images": val_images, "labels": val_labels},
                "test": {
                    "images": images[test_mask.to_numpy()],
                    "labels": labels[test_mask.to_numpy()],
                },
            }

    train_images, temp_images, train_labels, temp_labels = _safe_train_test_split(
        images,
        labels,
        test_size=val_ratio + test_ratio,
        random_state=random_seed,
        stratify=labels,
    )
    val_share = val_ratio / (val_ratio + test_ratio)
    val_images, test_images, val_labels, test_labels = _safe_train_test_split(
        temp_images,
        temp_labels,
        train_size=val_share,
        random_state=random_seed,
        stratify=temp_labels,
    )
    return {
        "train": {"images": train_images, "labels": train_labels},
        "val": {"images": val_images, "labels": val_labels},
        "test": {"images": test_images, "labels": test_labels},
    }


def _load_image_folder_splits(
    raw_dir: Path,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
    split_dirs = _discover_split_dirs(raw_dir)
    if split_dirs:
        class_names, class_to_index = _build_class_mapping(
            class_dir.name
            for split_dir in split_dirs.values()
            for class_dir in split_dir.iterdir()
            if class_dir.is_dir()
        )
        splits = {
            split_name: _load_image_split(split_dir, class_to_index)
            for split_name, split_dir in split_dirs.items()
        }
        if "train" not in splits:
            raise ValueError(
                "FER2013 image-folder layout must include a train/ directory "
                "when split directories are used."
            )
        if "val" not in splits and "test" not in splits:
            train_data = splits["train"]
            generated_splits = _split_arrays(
                images=train_data["images"],
                labels=train_data["labels"],
                usage=None,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                random_seed=random_seed,
            )
            splits.update(generated_splits)
        elif "val" not in splits:
            train_data = splits["train"]
            train_images, val_images, train_labels, val_labels = _safe_train_test_split(
                train_data["images"],
                train_data["labels"],
                test_size=val_ratio,
                random_state=random_seed,
                stratify=train_data["labels"],
            )
            splits["train"] = {"images": train_images, "labels": train_labels}
            splits["val"] = {"images": val_images, "labels": val_labels}
        elif "test" not in splits:
            train_data = splits["train"]
            train_images, test_images, train_labels, test_labels = _safe_train_test_split(
                train_data["images"],
                train_data["labels"],
                test_size=test_ratio,
                random_state=random_seed,
                stratify=train_data["labels"],
            )
            splits["train"] = {"images": train_images, "labels": train_labels}
            splits["test"] = {"images": test_images, "labels": test_labels}
        return splits, class_names

    class_names, class_to_index = _build_class_mapping(
        class_dir.name for class_dir in raw_dir.iterdir() if class_dir.is_dir()
    )
    full_data = _load_image_split(raw_dir, class_to_index)
    splits = _split_arrays(
        images=full_data["images"],
        labels=full_data["labels"],
        usage=None,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )
    return splits, class_names


def _discover_split_dirs(raw_dir: Path) -> dict[str, Path]:
    aliases = {
        "train": {"train", "training"},
        "val": {"val", "valid", "validation", "publictest"},
        "test": {"test", "privatetest", "private_test"},
    }
    split_dirs: dict[str, Path] = {}
    for child in raw_dir.iterdir():
        if not child.is_dir():
            continue
        normalized = child.name.lower().replace(" ", "").replace("-", "_")
        for split_name, names in aliases.items():
            if normalized in names:
                split_dirs[split_name] = child
                break
    return split_dirs


def _build_class_mapping(class_names: Any) -> tuple[list[str], dict[str, int]]:
    names = sorted({str(name) for name in class_names})
    if not names:
        raise ValueError("No class folders found in FER2013 image directory.")

    normalized_names = [name.lower().replace(" ", "").replace("-", "_") for name in names]
    if all(name in CLASS_NAME_ALIASES for name in normalized_names):
        class_to_index = {
            original: CLASS_NAME_ALIASES[normalized]
            for original, normalized in zip(names, normalized_names)
        }
        return EMOTION_LABELS, class_to_index

    return names, {name: index for index, name in enumerate(names)}


def _load_image_split(split_dir: Path, class_to_index: dict[str, int]) -> dict[str, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[int] = []

    for class_dir in sorted(child for child in split_dir.iterdir() if child.is_dir()):
        if class_dir.name not in class_to_index:
            continue
        label = class_to_index[class_dir.name]
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            images.append(_load_grayscale_image(image_path))
            labels.append(label)

    if not images:
        raise ValueError(f"No image files found under {split_dir}.")

    return {
        "images": np.stack(images).astype(np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
    }


def _load_grayscale_image(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        return image_array[..., np.newaxis].astype(np.float32)


def _ensure_channel_last(images: np.ndarray) -> np.ndarray:
    if images.ndim == 3:
        return images[..., np.newaxis].astype(np.float32)
    if images.ndim == 4 and images.shape[-1] == 1:
        return images.astype(np.float32)
    raise ValueError(f"Expected image array shape Nx48x48 or Nx48x48x1; got {images.shape}.")


def _safe_train_test_split(
    images: np.ndarray,
    labels: np.ndarray,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    requested_stratify = kwargs.pop("stratify", labels)
    try:
        return train_test_split(images, labels, stratify=requested_stratify, **kwargs)
    except ValueError:
        return train_test_split(images, labels, stratify=None, **kwargs)


def _count_images_by_class(directory: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for class_dir in sorted(child for child in directory.iterdir() if child.is_dir()):
        count = sum(
            1
            for image_path in class_dir.rglob("*")
            if image_path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if count > 0:
            counts[class_dir.name] = count
    return counts


def _class_counts(labels: np.ndarray, class_names: list[str]) -> dict[str, int]:
    counts = {class_name: 0 for class_name in class_names}
    unique_labels, unique_counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, unique_counts):
        counts[class_names[int(label)]] = int(count)
    return counts
