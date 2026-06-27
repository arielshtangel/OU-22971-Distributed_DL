"""Prepare deterministic metadata for the synthetic SimCLR-like workload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_SIZE = (3, 224, 224)
NUM_CLASSES = 1000
AUGMENTATIONS = [
    "RandomResizedCrop(224)",
    "RandomHorizontalFlip",
    "ColorJitter",
    "RandomGrayscale",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write metadata for the deterministic FakeData workload."
    )
    parser.add_argument("--dataset-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=22971)
    parser.add_argument("--output-dir", type=str, default="prepared")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset_size < 1:
        raise SystemExit("--dataset-size must be at least 1.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "dataset_metadata.json"

    metadata = {
        "dataset": "torchvision.datasets.FakeData",
        "dataset_size": args.dataset_size,
        "seed": args.seed,
        "image_size": list(IMAGE_SIZE),
        "num_classes": NUM_CLASSES,
        "labels_used": False,
        "positive_pair_views_per_source_image": 2,
        "augmentations": AUGMENTATIONS,
        "note": (
            "FakeData is generated lazily and deterministically from the seed; "
            "this file records the workload configuration, not downloaded data."
        ),
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {metadata_path}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
