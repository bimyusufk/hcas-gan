"""Mask-area curriculum sampler."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Sampler, Subset

from .dataset_loader import LabelMeCamouflageDataset


@dataclass
class BucketWeights:
    small: float
    medium: float
    large: float


def _polygon_area(points: np.ndarray) -> float:
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        return 0.0
    return float(abs(cv2.contourArea(points.astype(np.float32))))


def _estimate_area_ratio(dataset: LabelMeCamouflageDataset, sample_index: int) -> float:
    record = dataset.records[sample_index]
    image = cv2.imread(str(record.image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0
    h, w = image.shape[:2]
    total = max(float(h * w), 1.0)
    area = 0.0
    for poly in record.polygons:
        area += _polygon_area(poly)
    return float(np.clip(area / total, 0.0, 1.0))


class MaskAreaCurriculumSampler(Sampler[int]):
    """Weighted sampler that biases samples by mask area buckets over epochs."""

    def __init__(
        self,
        *,
        train_subset: Subset,
        small_threshold: float = 0.08,
        medium_threshold: float = 0.2,
        start_weights: tuple[float, float, float] = (0.2, 0.5, 0.3),
        end_weights: tuple[float, float, float] = (0.1, 0.3, 0.6),
        curriculum_epochs: int = 100,
        seed: int = 42,
    ):
        if not isinstance(train_subset, Subset):
            raise TypeError("MaskAreaCurriculumSampler expects a torch.utils.data.Subset for train_subset.")
        if not isinstance(train_subset.dataset, LabelMeCamouflageDataset):
            raise TypeError("Subset dataset must be LabelMeCamouflageDataset.")

        self.subset = train_subset
        self.dataset = train_subset.dataset
        self.indices = list(train_subset.indices)
        self.small_threshold = float(small_threshold)
        self.medium_threshold = float(medium_threshold)
        self.start_weights = BucketWeights(*[float(x) for x in start_weights])
        self.end_weights = BucketWeights(*[float(x) for x in end_weights])
        self.curriculum_epochs = max(1, int(curriculum_epochs))
        self.seed = int(seed)
        self.epoch = 1

        self._bucket_ids: list[int] = []
        self._areas: list[float] = []
        self._weights = torch.ones(len(self.indices), dtype=torch.float32)
        self._build_buckets()
        self._refresh_weights()

    def _bucket_for_ratio(self, ratio: float) -> int:
        if ratio < self.small_threshold:
            return 0  # small
        if ratio < self.medium_threshold:
            return 1  # medium
        return 2  # large

    def _build_buckets(self) -> None:
        self._bucket_ids.clear()
        self._areas.clear()
        for idx in self.indices:
            ratio = _estimate_area_ratio(self.dataset, idx)
            self._areas.append(ratio)
            self._bucket_ids.append(self._bucket_for_ratio(ratio))

    def _interp_bucket_weights(self) -> BucketWeights:
        t = min(float(max(1, self.epoch)) / float(self.curriculum_epochs), 1.0)
        return BucketWeights(
            small=self.start_weights.small + (self.end_weights.small - self.start_weights.small) * t,
            medium=self.start_weights.medium + (self.end_weights.medium - self.start_weights.medium) * t,
            large=self.start_weights.large + (self.end_weights.large - self.start_weights.large) * t,
        )

    def _refresh_weights(self) -> None:
        bw = self._interp_bucket_weights()
        weights = []
        for bucket_id in self._bucket_ids:
            if bucket_id == 0:
                weights.append(max(bw.small, 1e-8))
            elif bucket_id == 1:
                weights.append(max(bw.medium, 1e-8))
            else:
                weights.append(max(bw.large, 1e-8))
        self._weights = torch.tensor(weights, dtype=torch.float32)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(1, int(epoch))
        self._refresh_weights()

    def get_bucket_stats(self) -> dict[str, float]:
        total = max(len(self._bucket_ids), 1)
        n_small = sum(1 for x in self._bucket_ids if x == 0)
        n_medium = sum(1 for x in self._bucket_ids if x == 1)
        n_large = sum(1 for x in self._bucket_ids if x == 2)
        return {
            "small_ratio": float(n_small / total),
            "medium_ratio": float(n_medium / total),
            "large_ratio": float(n_large / total),
        }

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        num_samples = len(self.indices)
        sampled_pos = torch.multinomial(self._weights, num_samples=num_samples, replacement=True, generator=generator)
        for pos in sampled_pos.tolist():
            # IMPORTANT: for DataLoader(dataset=Subset(...), sampler=...), sampler must
            # yield indices relative to the Subset (0..len(subset)-1), not base dataset.
            yield int(pos)

    def __len__(self) -> int:
        return len(self.indices)
