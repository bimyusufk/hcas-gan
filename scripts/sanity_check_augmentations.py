"""Sanity check for train-time augmentation pipeline.

Validates:
1) Augmenter builds from config and is callable.
2) Collate with augmenter returns valid tensor shapes/ranges.
3) Augmented masks remain non-empty and binary.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import DatasetConfig, LabelMeCamouflageDataset, build_augmentations, build_resize_collate_fn


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    _set_seed(seed)

    dataset_cfg = DatasetConfig(
        annotations_dir=str((PROJECT_ROOT / cfg["data"]["annotations_dir"]).resolve()),
        image_suffix=str(cfg["data"].get("image_suffix", ".png")),
        annotation_suffix=str(cfg["data"].get("annotation_suffix", ".json")),
        target_label=str(cfg["data"].get("target_label", "camou")),
        ignore_labels=tuple(cfg["data"].get("ignore_labels", ["skin"])),
        strict_non_empty_mask=True,
    )
    dataset = LabelMeCamouflageDataset(dataset_cfg)

    n_subset = min(16, len(dataset))
    subset = Subset(dataset, list(range(n_subset)))

    augmenter = build_augmentations(cfg.get("augmentations", {}))
    if augmenter is None:
        raise RuntimeError("Augmentations are disabled in config; cannot run augmentation sanity check.")

    strength_start = None
    strength_mid = None
    strength_end = None
    if hasattr(augmenter, "set_epoch") and hasattr(augmenter, "get_strength_factor"):
        augmenter.set_epoch(1)
        strength_start = float(augmenter.get_strength_factor())
        augmenter.set_epoch(max(2, int(cfg.get("augmentations", {}).get("schedule_warmup_epochs", 20)) // 2))
        strength_mid = float(augmenter.get_strength_factor())
        augmenter.set_epoch(int(cfg.get("augmentations", {}).get("schedule_warmup_epochs", 20)))
        strength_end = float(augmenter.get_strength_factor())
        if not (strength_start <= strength_mid <= strength_end):
            raise AssertionError(
                "Augmentation schedule factor should be non-decreasing during warmup. "
                f"Got start={strength_start}, mid={strength_mid}, end={strength_end}."
            )

    image_size = int(cfg.get("hcas_specific", {}).get("image_size", 256))
    collate_fn = build_resize_collate_fn(image_size=image_size, augmenter=augmenter)

    loader = DataLoader(
        subset,
        batch_size=min(int(cfg.get("hyperparameters", {}).get("batch_size", 4)), 4),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    batch = next(iter(loader))
    images = batch["image"]
    masks = batch["mask"]

    assert isinstance(images, torch.Tensor) and isinstance(masks, torch.Tensor)
    assert images.ndim == 4 and images.shape[1] == 3
    assert masks.ndim == 4 and masks.shape[1] == 1
    assert images.shape[-2:] == masks.shape[-2:] == (image_size, image_size)

    assert torch.isfinite(images).all(), "Augmented images contain non-finite values"
    assert torch.isfinite(masks).all(), "Augmented masks contain non-finite values"

    assert float(images.min().item()) >= 0.0 and float(images.max().item()) <= 1.0, (
        "Augmented image values must stay within [0,1]"
    )

    # Binary-like check after threshold in collate_fn.
    unique_mask_vals = torch.unique(masks)
    assert all(v in (0.0, 1.0) for v in unique_mask_vals.tolist()), (
        f"Mask contains non-binary values: {unique_mask_vals.tolist()}"
    )

    # Non-empty masks are required by saliency penalty.
    mask_sums = torch.sum(masks, dim=(1, 2, 3))
    assert bool(torch.all(mask_sums > 0)), "At least one augmented mask became empty"

    print(f"subset_size={n_subset}")
    print(f"batch_image_shape={tuple(images.shape)}")
    print(f"batch_mask_shape={tuple(masks.shape)}")
    print(f"mask_pixels_per_sample={mask_sums.tolist()}")
    if strength_start is not None:
        print(
            f"schedule_strength=start:{strength_start:.3f} "
            f"mid:{strength_mid:.3f} end:{strength_end:.3f}"
        )
    print("AUGMENTATION SANITY CHECK PASSED")


if __name__ == "__main__":
    main()
