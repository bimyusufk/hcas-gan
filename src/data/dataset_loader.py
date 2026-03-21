"""Dataset loader for LabelMe polygon annotations.

This module targets the camouflage dataset structure where each `.json`
annotation has a paired image file (`.png`) in the same directory.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
from typing import Any, Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset


def _read_json_relaxed(json_path: Path) -> dict[str, Any]:
    """Load JSON file with fallback for trailing commas.

    Some LabelMe exports in this dataset include trailing commas before
    `}` or `]`, which is invalid strict JSON. We sanitize those patterns
    if strict parsing fails.
    """
    raw = json_path.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
        return json.loads(cleaned)


def _normalize_label(label: str) -> str:
    return label.strip().lower()


def _resolve_image_path(
    annotation_path: Path,
    image_path_field: str | None,
    image_suffix: str,
) -> Path:
    """Resolve image path from LabelMe field with robust fallbacks."""
    candidates: list[Path] = []

    if image_path_field:
        field_as_path = Path(image_path_field)
        candidates.append((annotation_path.parent / field_as_path).resolve())
        candidates.append((annotation_path.parent / field_as_path.name).resolve())

    candidates.append(annotation_path.with_suffix(image_suffix).resolve())

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Image pair not found for annotation: {annotation_path.name}"
    )


def _polygon_to_int_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert float polygon points to clipped int32 OpenCV points."""
    pts = np.round(points).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    return pts


@dataclass
class DatasetConfig:
    annotations_dir: str
    image_suffix: str = ".png"
    annotation_suffix: str = ".json"
    target_label: str = "camou"
    ignore_labels: tuple[str, ...] = ("skin",)
    strict_non_empty_mask: bool = True
    min_polygon_points: int = 3
    normalize_image: bool = True


@dataclass
class SampleRecord:
    annotation_path: Path
    image_path: Path
    polygons: list[np.ndarray]


class LabelMeCamouflageDataset(Dataset[dict[str, torch.Tensor | str]]):
    """PyTorch dataset for LabelMe polygon-based camouflage masks."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.annotations_dir = Path(config.annotations_dir).resolve()
        if not self.annotations_dir.exists():
            raise FileNotFoundError(
                f"Annotations directory not found: {self.annotations_dir}"
            )

        self.target_label = _normalize_label(config.target_label)
        self.ignore_labels = {_normalize_label(x) for x in config.ignore_labels}
        self.records: list[SampleRecord] = []

        self._build_index()

        if not self.records:
            raise RuntimeError(
                "No valid samples found after indexing annotations. "
                "Check label names/path settings in DatasetConfig."
            )

    def _build_index(self) -> None:
        ann_files = sorted(self.annotations_dir.glob(f"*{self.config.annotation_suffix}"))
        for ann_path in ann_files:
            try:
                data = _read_json_relaxed(ann_path)
                image_path = _resolve_image_path(
                    annotation_path=ann_path,
                    image_path_field=data.get("imagePath"),
                    image_suffix=self.config.image_suffix,
                )
                polygons = self._extract_target_polygons(data)

                if self.config.strict_non_empty_mask and len(polygons) == 0:
                    continue

                self.records.append(
                    SampleRecord(
                        annotation_path=ann_path,
                        image_path=image_path,
                        polygons=polygons,
                    )
                )
            except Exception:
                # Skip malformed pairs; use sanity script to inspect dataset quality.
                continue

    def _extract_target_polygons(self, data: dict[str, Any]) -> list[np.ndarray]:
        polygons: list[np.ndarray] = []
        shapes = data.get("shapes", [])
        if not isinstance(shapes, list):
            return polygons

        for shape in shapes:
            if not isinstance(shape, dict):
                continue

            label = _normalize_label(str(shape.get("label", "")))
            if label in self.ignore_labels:
                continue
            if label != self.target_label:
                continue

            if shape.get("shape_type", "polygon") != "polygon":
                continue

            points = shape.get("points", [])
            if not isinstance(points, list):
                continue

            pts = np.asarray(points, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] != 2:
                continue
            if pts.shape[0] < self.config.min_polygon_points:
                continue

            polygons.append(pts)

        return polygons

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        record = self.records[idx]

        image_bgr = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {record.image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in record.polygons:
            pts = _polygon_to_int_points(poly, width=w, height=h)
            cv2.fillPoly(mask, [pts], 1)

        if self.config.strict_non_empty_mask and int(mask.sum()) == 0:
            raise RuntimeError(
                f"Generated empty mask for sample: {record.annotation_path.name}"
            )

        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float()
        if self.config.normalize_image:
            image_tensor = image_tensor / 255.0

        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(record.image_path),
            "annotation_path": str(record.annotation_path),
        }


def reproducible_split_indices(
    n_samples: int,
    *,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Generate deterministic train/val/test indices."""
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if not (0.0 <= val_ratio < 1.0 and 0.0 <= test_ratio < 1.0):
        raise ValueError("val_ratio and test_ratio must be within [0, 1)")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1")

    indices = list(range(n_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_test = int(n_samples * test_ratio)
    n_val = int(n_samples * val_ratio)

    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]
    return train_idx, val_idx, test_idx


def split_dataset(
    dataset: Dataset[dict[str, torch.Tensor | str]],
    *,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """Split dataset into train/val/test subsets deterministically."""
    train_idx, val_idx, test_idx = reproducible_split_indices(
        len(dataset), val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


def build_resize_collate_fn(
    image_size: int | tuple[int, int],
    augmenter: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> Any:
    """Build a collate function that resizes image+mask then stacks tensors.

    Useful for datasets with mixed native resolutions.
    """
    if isinstance(image_size, int):
        target_hw = (image_size, image_size)
    else:
        target_hw = image_size

    if len(target_hw) != 2:
        raise ValueError(f"image_size must be int or tuple(H,W), got: {image_size}")

    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Target image size must be > 0, got: {(target_h, target_w)}")

    def _collate(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, Any]:
        if not batch:
            raise ValueError("Cannot collate empty batch.")

        images: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        image_paths: list[str] = []
        annotation_paths: list[str] = []

        for sample in batch:
            image = sample.get("image")
            mask = sample.get("mask")
            if not isinstance(image, torch.Tensor) or not isinstance(mask, torch.Tensor):
                raise TypeError("Each sample must contain torch.Tensor 'image' and 'mask'.")

            if image.ndim != 3 or mask.ndim != 3:
                raise ValueError(
                    f"Expected image/mask to be [C,H,W], got {image.shape} / {mask.shape}."
                )

            image_rs = F.interpolate(
                image.unsqueeze(0),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            mask_rs = F.interpolate(
                mask.unsqueeze(0),
                size=(target_h, target_w),
                mode="nearest",
            ).squeeze(0)
            mask_rs = (mask_rs > 0.5).float()

            original_mask_rs = mask_rs

            if augmenter is not None:
                image_rs, mask_rs = augmenter(image_rs, mask_rs)
                if not isinstance(image_rs, torch.Tensor) or not isinstance(mask_rs, torch.Tensor):
                    raise TypeError("Augmenter must return (torch.Tensor image, torch.Tensor mask).")
                if image_rs.shape != image.shape[:1] + (target_h, target_w):
                    raise ValueError(
                        "Augmented image must keep shape [3,H,W]. "
                        f"Got {tuple(image_rs.shape)}, expected {(image.shape[0], target_h, target_w)}"
                    )
                if mask_rs.shape != mask.shape[:1] + (target_h, target_w):
                    raise ValueError(
                        "Augmented mask must keep shape [1,H,W]. "
                        f"Got {tuple(mask_rs.shape)}, expected {(mask.shape[0], target_h, target_w)}"
                    )

            mask_rs = (mask_rs > 0.5).float()
            if torch.sum(mask_rs) <= 0:
                mask_rs = original_mask_rs
            image_rs = image_rs.clamp(0.0, 1.0)

            images.append(image_rs)
            masks.append(mask_rs)
            image_paths.append(str(sample.get("image_path", "")))
            annotation_paths.append(str(sample.get("annotation_path", "")))

        return {
            "image": torch.stack(images, dim=0),
            "mask": torch.stack(masks, dim=0),
            "image_path": image_paths,
            "annotation_path": annotation_paths,
        }

    return _collate

