"""Data augmentation pipeline for HCAS-GAN.

This module applies train-time augmentations while preserving image/mask
consistency requirements:
- photometric transforms are applied to image only
- geometric transforms for placement/shape are applied to mask safely
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def _uniform(min_v: float, max_v: float) -> float:
    if max_v <= min_v:
        return float(min_v)
    return float(min_v + (max_v - min_v) * torch.rand(1).item())


@dataclass
class CamouflageAugmentationPipeline:
    """Train-time augmentation pipeline for image/mask pairs."""

    random_mask_rotation: bool = True
    max_rotation_deg: float = 25.0
    mask_scale_range: tuple[float, float] = (0.3, 0.8)
    max_translate_ratio: float = 0.08

    hflip_prob: float = 0.5

    color_jitter: bool = True
    color_jitter_prob: float = 0.7
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.15
    hue: float = 0.02

    # Optional augmentation intensity schedule
    schedule_enabled: bool = False
    schedule_start_factor: float = 0.35
    schedule_end_factor: float = 1.0
    schedule_warmup_epochs: int = 20

    current_epoch: int = 1

    def __post_init__(self) -> None:
        min_scale, max_scale = self.mask_scale_range
        if min_scale <= 0 or max_scale <= 0:
            raise ValueError("mask_scale_range values must be > 0")
        if min_scale > max_scale:
            raise ValueError("mask_scale_range must satisfy min <= max")

        if self.max_rotation_deg < 0:
            raise ValueError("max_rotation_deg must be >= 0")
        if not (0.0 <= self.max_translate_ratio <= 1.0):
            raise ValueError("max_translate_ratio must be within [0,1]")

        if not (0.0 <= self.hflip_prob <= 1.0):
            raise ValueError("hflip_prob must be within [0,1]")
        if not (0.0 <= self.color_jitter_prob <= 1.0):
            raise ValueError("color_jitter_prob must be within [0,1]")

        if self.brightness < 0 or self.contrast < 0 or self.saturation < 0:
            raise ValueError("brightness/contrast/saturation must be >= 0")
        if self.hue < 0 or self.hue > 0.5:
            raise ValueError("hue must be within [0, 0.5]")

        if self.schedule_start_factor < 0.0 or self.schedule_end_factor < 0.0:
            raise ValueError("schedule_start_factor/schedule_end_factor must be >= 0")
        if self.schedule_warmup_epochs <= 0:
            raise ValueError("schedule_warmup_epochs must be > 0")

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for schedule-based augmentation intensity."""
        self.current_epoch = max(1, int(epoch))

    def get_strength_factor(self) -> float:
        """Get current augmentation intensity multiplier."""
        if not self.schedule_enabled:
            return 1.0

        if self.schedule_warmup_epochs <= 1:
            return float(self.schedule_end_factor)

        t = (self.current_epoch - 1) / float(self.schedule_warmup_epochs - 1)
        t = max(0.0, min(1.0, t))
        return float(self.schedule_start_factor + (self.schedule_end_factor - self.schedule_start_factor) * t)

    def _scaled_mask_scale_range(self, strength: float) -> tuple[float, float]:
        base_min, base_max = self.mask_scale_range
        min_scale = 1.0 + strength * (base_min - 1.0)
        max_scale = 1.0 + strength * (base_max - 1.0)
        min_scale = max(1e-4, float(min_scale))
        max_scale = max(1e-4, float(max_scale))
        if min_scale > max_scale:
            min_scale, max_scale = max_scale, min_scale
        return min_scale, max_scale

    def _augment_mask(self, mask: torch.Tensor, strength: float) -> torch.Tensor:
        out = mask
        max_rotation_deg = self.max_rotation_deg * strength
        max_translate_ratio = self.max_translate_ratio * strength

        if self.random_mask_rotation and max_rotation_deg > 0:
            angle = _uniform(-max_rotation_deg, max_rotation_deg)
            out = TF.rotate(
                out,
                angle=angle,
                interpolation=InterpolationMode.NEAREST,
                fill=0.0,
            )

        min_scale, max_scale = self._scaled_mask_scale_range(strength)
        if min_scale != 1.0 or max_scale != 1.0:
            scale = _uniform(min_scale, max_scale)
            out = TF.affine(
                out,
                angle=0.0,
                translate=[0, 0],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0.0,
            )

        if max_translate_ratio > 0.0:
            h, w = int(out.shape[-2]), int(out.shape[-1])
            tx = int(round(_uniform(-max_translate_ratio, max_translate_ratio) * w))
            ty = int(round(_uniform(-max_translate_ratio, max_translate_ratio) * h))
            out = TF.affine(
                out,
                angle=0.0,
                translate=[tx, ty],
                scale=1.0,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0.0,
            )

        return (out > 0.5).float()

    def _augment_image_photometric(self, image: torch.Tensor, strength: float) -> torch.Tensor:
        out = image
        jitter_prob = self.color_jitter_prob * strength
        brightness = self.brightness * strength
        contrast = self.contrast * strength
        saturation = self.saturation * strength
        hue = self.hue * strength

        if self.color_jitter and torch.rand(1).item() < jitter_prob:
            if brightness > 0:
                factor = _uniform(max(0.0, 1.0 - brightness), 1.0 + brightness)
                out = TF.adjust_brightness(out, factor)
            if contrast > 0:
                factor = _uniform(max(0.0, 1.0 - contrast), 1.0 + contrast)
                out = TF.adjust_contrast(out, factor)
            if saturation > 0:
                factor = _uniform(max(0.0, 1.0 - saturation), 1.0 + saturation)
                out = TF.adjust_saturation(out, factor)
            if hue > 0:
                factor = _uniform(-hue, hue)
                out = TF.adjust_hue(out, factor)

        return out.clamp(0.0, 1.0)

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"image must be [3,H,W], got: {tuple(image.shape)}")
        if mask.ndim != 3 or mask.shape[0] != 1:
            raise ValueError(f"mask must be [1,H,W], got: {tuple(mask.shape)}")
        if image.shape[-2:] != mask.shape[-2:]:
            raise ValueError(
                "image/mask spatial shape mismatch: "
                f"{tuple(image.shape[-2:])} vs {tuple(mask.shape[-2:])}"
            )

        original_mask = (mask > 0.5).float()
        strength = self.get_strength_factor()

        out_image = image
        out_mask = self._augment_mask(mask, strength)

        hflip_prob = self.hflip_prob * (0.5 + 0.5 * strength)
        if torch.rand(1).item() < hflip_prob:
            out_image = TF.hflip(out_image)
            out_mask = TF.hflip(out_mask)

        out_image = self._augment_image_photometric(out_image, strength)

        # Keep mask non-empty for stable saliency penalty computation.
        if torch.sum(out_mask) <= 0:
            out_mask = original_mask

        return out_image, out_mask


def build_augmentations(config: dict[str, Any]) -> Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None:
    """Build train augmentation callable from config.

    Returns:
        callable(image, mask) -> (image, mask), or None when disabled.
    """
    if not config:
        return None

    enabled = bool(config.get("enabled", True))
    if not enabled:
        return None

    mask_scale_cfg = config.get("mask_scale_range", [1.0, 1.0])
    if not isinstance(mask_scale_cfg, (list, tuple)) or len(mask_scale_cfg) != 2:
        raise ValueError("augmentations.mask_scale_range must be [min,max]")

    pipeline = CamouflageAugmentationPipeline(
        random_mask_rotation=bool(config.get("random_mask_rotation", True)),
        max_rotation_deg=float(config.get("max_rotation_deg", 25.0)),
        mask_scale_range=(float(mask_scale_cfg[0]), float(mask_scale_cfg[1])),
        max_translate_ratio=float(config.get("max_translate_ratio", 0.08)),
        hflip_prob=float(config.get("hflip_prob", 0.5)),
        color_jitter=bool(config.get("color_jitter", True)),
        color_jitter_prob=float(config.get("color_jitter_prob", 0.7)),
        brightness=float(config.get("brightness", 0.2)),
        contrast=float(config.get("contrast", 0.2)),
        saturation=float(config.get("saturation", 0.15)),
        hue=float(config.get("hue", 0.02)),
        schedule_enabled=bool(config.get("schedule_enabled", False)),
        schedule_start_factor=float(config.get("schedule_start_factor", 0.35)),
        schedule_end_factor=float(config.get("schedule_end_factor", 1.0)),
        schedule_warmup_epochs=int(config.get("schedule_warmup_epochs", 20)),
    )

    return pipeline
