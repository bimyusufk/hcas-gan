"""Frequency-domain regularization loss."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn


def _to_gray(x: torch.Tensor) -> torch.Tensor:
    # x: [B,3,H,W]
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _radial_profile(power: torch.Tensor, n_bins: int) -> torch.Tensor:
    # power: [B,H,W], values >=0
    b, h, w = power.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=power.device),
        torch.arange(w, device=power.device),
        indexing="ij",
    )
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    rr = torch.sqrt((yy.float() - cy) ** 2 + (xx.float() - cx) ** 2)
    max_r = float(rr.max().item()) + 1e-8
    rr_bin = torch.clamp((rr / max_r * (n_bins - 1)).long(), 0, n_bins - 1)

    profiles: list[torch.Tensor] = []
    for i in range(b):
        vals = power[i]
        sums = torch.zeros(n_bins, device=power.device, dtype=power.dtype)
        counts = torch.zeros(n_bins, device=power.device, dtype=power.dtype)
        sums.scatter_add_(0, rr_bin.reshape(-1), vals.reshape(-1))
        counts.scatter_add_(0, rr_bin.reshape(-1), torch.ones_like(vals).reshape(-1))
        profile = sums / torch.clamp_min(counts, 1.0)
        profiles.append(profile)
    return torch.stack(profiles, dim=0)  # [B,n_bins]


def _spectrum_profile(gray: torch.Tensor, n_bins: int) -> torch.Tensor:
    # gray: [B,1,H,W]
    b = gray.shape[0]
    x = gray.squeeze(1)
    fft = torch.fft.fft2(x)
    fft = torch.fft.fftshift(fft, dim=(-2, -1))
    power = (fft.real ** 2 + fft.imag ** 2).float()
    profile = _radial_profile(power, n_bins=n_bins)
    profile = profile / torch.clamp_min(profile.mean(dim=1, keepdim=True), 1e-8)
    return profile


def _estimate_target_profile(dataset_dir: Path, image_size: int, n_bins: int, max_images: int = 200) -> np.ndarray | None:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = [p for p in sorted(dataset_dir.rglob("*")) if p.is_file() and p.suffix.lower() in exts][:max_images]
    if not files:
        return None

    profiles = []
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
        x = img.astype(np.float32) / 255.0
        fft = np.fft.fftshift(np.fft.fft2(x))
        power = np.abs(fft) ** 2

        h, w = power.shape
        yy, xx = np.indices((h, w))
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        rr_bin = np.clip((rr / (rr.max() + 1e-8) * (n_bins - 1)).astype(np.int32), 0, n_bins - 1)

        sums = np.bincount(rr_bin.reshape(-1), weights=power.reshape(-1), minlength=n_bins).astype(np.float32)
        counts = np.bincount(rr_bin.reshape(-1), minlength=n_bins).astype(np.float32)
        profile = sums / np.clip(counts, 1.0, None)
        profile = profile / max(float(profile.mean()), 1e-8)
        profiles.append(profile)

    if not profiles:
        return None
    return np.mean(np.stack(profiles, axis=0), axis=0).astype(np.float32)


class FrequencyLoss(nn.Module):
    """Match radial frequency profile of generated patterns to style references."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        dataset_dir: str | None = None,
        image_size: int = 256,
        n_bins: int = 32,
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.dataset_dir = dataset_dir
        self.image_size = int(image_size)
        self.n_bins = int(n_bins)

        profile = None
        if self.enabled and self.dataset_dir:
            d = Path(self.dataset_dir).resolve()
            if d.exists():
                profile = _estimate_target_profile(d, image_size=self.image_size, n_bins=self.n_bins)

        if profile is None:
            self.enabled = False
            self.register_buffer("target_profile", torch.empty(0), persistent=False)
        else:
            self.register_buffer("target_profile", torch.from_numpy(profile), persistent=False)

    @property
    def is_ready(self) -> bool:
        return bool(self.enabled and self.target_profile.ndim == 1 and self.target_profile.shape[0] > 0)

    def forward(self, fake_pattern: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if not self.is_ready:
            return torch.zeros((), device=fake_pattern.device, dtype=fake_pattern.dtype)

        x = fake_pattern
        if mask is not None:
            x = x * mask

        gray = _to_gray(x)
        pred_profile = _spectrum_profile(gray, n_bins=self.n_bins)
        target = self.target_profile.to(device=fake_pattern.device, dtype=fake_pattern.dtype).unsqueeze(0)
        target = target.expand(pred_profile.shape[0], -1)
        return torch.mean(torch.abs(pred_profile - target))
