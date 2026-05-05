"""Prepare FER2013 data for model training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare FER2013 train/val/test splits.")
    parser.add_argument(
        "--raw-dir",
        "--dataset-path",
        dest="raw_dir",
        default="data/raw/fer2013",
        help="Raw FER2013 directory.",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed/fer2013",
        help="Output directory for prepared .npz files.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=466, help="Random seed for generated splits.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing prepared files.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the raw FER2013 layout without writing processed arrays.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Create a tiny synthetic FER2013-like processed dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.data.fer2013_loader import (
        create_synthetic_fer2013,
        prepare_fer2013,
        validate_fer2013_raw,
    )

    if args.validate_only:
        summary = validate_fer2013_raw(args.raw_dir)
        print("FER2013 raw dataset validation complete.")
        print(json.dumps(summary, indent=2))
        return

    if args.smoke_test:
        metadata = create_synthetic_fer2013(
            processed_dir=args.processed_dir,
            random_seed=args.seed,
            force=True,
        )
        print("Synthetic FER2013 smoke dataset created.")
        print(json.dumps(metadata["split_counts"], indent=2))
        return

    metadata = prepare_fer2013(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        force=args.force,
    )
    print("FER2013 preparation complete.")
    print(json.dumps(metadata["split_counts"], indent=2))


if __name__ == "__main__":
    main()
