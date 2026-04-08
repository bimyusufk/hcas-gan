"""Style-prior loss utilities based on Gram statistics.

This module keeps implementation lightweight (no external pretrained backbone required)
by computing Gram statistics directly on RGB feature maps at multiple scales.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import torch
import torch.nn.functional as F
from torch import nn


def _list_image_files(directory: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted([p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def _read_image_tensor(path: Path, image_size: int) -> torch.Tensor | None:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    return tensor


def _gram_matrix(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # x: [B,C,H,W]
    b, c, h, w = x.shape
    feats = x.view(b, c, h * w)
    gram = torch.bmm(feats, feats.transpose(1, 2))
    denom = float(c * h * w)
    return gram / max(denom, eps)


class StylePriorLoss(nn.Module):
    """Gram-based style prior loss against a cached style reference bank."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        dataset_dir: str | None = None,
        image_size: int = 256,
        max_images: int = 256,
        multiscale: tuple[float, ...] = (1.0, 0.5),
        eps: float = 1e-8,
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.dataset_dir = str(dataset_dir) if dataset_dir else ""
        self.image_size = int(image_size)
        self.max_images = int(max_images)
        self.multiscale = tuple(float(s) for s in multiscale if s > 0)
        self.eps = float(eps)

        self.register_buffer("style_bank", torch.empty(0), persistent=False)

        if self.enabled and self.dataset_dir:
            self._load_style_bank()

    @property
    def is_ready(self) -> bool:
        return bool(self.enabled and self.style_bank.ndim == 4 and self.style_bank.shape[0] > 0)

    def _load_style_bank(self) -> None:
        directory = Path(self.dataset_dir).resolve()
        if not directory.exists():
            self.enabled = False
            return

        files = _list_image_files(directory)
        if not files:
            self.enabled = False
            return

        tensors: list[torch.Tensor] = []
        for p in files[: self.max_images]:
            t = _read_image_tensor(p, image_size=self.image_size)
            if t is not None:
                tensors.append(t)

        if not tensors:
            self.enabled = False
            return

        bank = torch.stack(tensors, dim=0)  # [N,3,H,W]
        self.style_bank = bank

    def _sample_style_batch(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.style_bank.shape[0] == 0:
            return torch.empty(0, device=device)
        idx = torch.randint(low=0, high=self.style_bank.shape[0], size=(batch_size,), device=device)
        return self.style_bank.to(device=device).index_select(0, idx)

    def _multiscale_gram_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        total = torch.zeros((), device=x.device, dtype=x.dtype)
        n_scales = 0

        for scale in self.multiscale:
            if abs(scale - 1.0) < 1e-6:
                x_s = x
                y_s = y
            else:
                h = max(8, int(x.shape[2] * scale))
                w = max(8, int(x.shape[3] * scale))
                x_s = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
                y_s = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)

            gram_x = _gram_matrix(x_s, eps=self.eps)
            gram_y = _gram_matrix(y_s, eps=self.eps)
            total = total + F.l1_loss(gram_x, gram_y)
            n_scales += 1

        return total / max(n_scales, 1)

    def forward(self, fake_pattern: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if not self.is_ready:
            return torch.zeros((), device=fake_pattern.device, dtype=fake_pattern.dtype)

        fake = fake_pattern
        if mask is not None:
            fake = fake * mask

        style_batch = self._sample_style_batch(fake.shape[0], device=fake.device)
        if style_batch.numel() == 0:
            return torch.zeros((), device=fake_pattern.device, dtype=fake_pattern.dtype)

        return self._multiscale_gram_loss(fake, style_batch)
