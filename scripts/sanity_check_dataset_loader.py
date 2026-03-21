"""Sanity check for LabelMe camouflage dataset loader."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_loader import DatasetConfig, LabelMeCamouflageDataset, split_dataset


def main() -> None:
    dataset = LabelMeCamouflageDataset(
        DatasetConfig(
            annotations_dir=str(PROJECT_ROOT / "camo"),
            target_label="camou",
            ignore_labels=("skin",),
            strict_non_empty_mask=True,
        )
    )

    train_set, val_set, test_set = split_dataset(dataset, val_ratio=0.1, test_ratio=0.1, seed=42)

    sample = dataset[0]
    image = sample["image"]
    mask = sample["mask"]

    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert image.ndim == 3 and image.shape[0] == 3
    assert mask.ndim == 3 and mask.shape[0] == 1
    assert torch.sum(mask) > 0, "Mask should not be empty"

    print(f"dataset_size={len(dataset)}")
    print(f"train_size={len(train_set)} val_size={len(val_set)} test_size={len(test_set)}")
    print(f"image_shape={tuple(image.shape)} mask_shape={tuple(mask.shape)}")
    print(f"sample_image={sample['image_path']}")
    print("DATASET SANITY CHECK PASSED")


if __name__ == "__main__":
    main()
