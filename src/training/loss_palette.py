"""Palette-constrained loss utilities."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn


def _kmeans_palette_from_dir(directory: Path, n_colors: int, max_images: int = 200) -> list[list[int]]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    image_files = [p for p in sorted(directory.rglob("*")) if p.is_file() and p.suffix.lower() in exts][:max_images]
    if not image_files:
        return []

    pixels: list[np.ndarray] = []
    for p in image_files:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        flat = rgb.reshape(-1, 3)
        if flat.shape[0] > 10000:
            idx = np.random.choice(flat.shape[0], size=10000, replace=False)
            flat = flat[idx]
        pixels.append(flat.astype(np.float32))

    if not pixels:
        return []

    sample = np.concatenate(pixels, axis=0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2)
    _, _, centers = cv2.kmeans(sample, n_colors, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    centers = np.clip(np.round(centers), 0, 255).astype(np.int32)
    return centers.tolist()


def _load_palette_colors(
    palette_json_path: str | None,
    dataset_dir: str | None,
    n_colors: int,
) -> list[list[int]]:
    if palette_json_path:
        p = Path(palette_json_path).resolve()
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                if "colors_rgb" in data and isinstance(data["colors_rgb"], list):
                    return data["colors_rgb"]
                if "palette" in data and isinstance(data["palette"], dict):
                    colors = data["palette"].get("colors_rgb")
                    if isinstance(colors, list):
                        return colors
                if "domains" in data and isinstance(data["domains"], dict):
                    merged: list[list[int]] = []
                    for domain_value in data["domains"].values():
                        if isinstance(domain_value, dict):
                            colors = domain_value.get("colors_rgb")
                            if isinstance(colors, list):
                                merged.extend(colors)
                    if merged:
                        return merged
            if isinstance(data, list):
                return data

    if dataset_dir:
        d = Path(dataset_dir).resolve()
        if d.exists():
            return _kmeans_palette_from_dir(d, n_colors=n_colors)

    return []


class PaletteLoss(nn.Module):
    """Distance-to-palette loss using nearest palette color assignment."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        palette_json_path: str | None = None,
        dataset_dir: str | None = None,
        n_colors: int = 6,
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.palette_json_path = palette_json_path
        self.dataset_dir = dataset_dir
        self.n_colors = int(n_colors)

        colors = _load_palette_colors(
            palette_json_path=self.palette_json_path,
            dataset_dir=self.dataset_dir,
            n_colors=self.n_colors,
        )

        if not colors:
            self.enabled = False
            self.register_buffer("palette", torch.empty(0), persistent=False)
        else:
            palette = torch.tensor(colors, dtype=torch.float32) / 255.0
            if palette.ndim != 2 or palette.shape[1] != 3:
                self.enabled = False
                self.register_buffer("palette", torch.empty(0), persistent=False)
            else:
                self.register_buffer("palette", palette, persistent=False)

    @property
    def is_ready(self) -> bool:
        return bool(self.enabled and self.palette.ndim == 2 and self.palette.shape[0] > 0)

    def forward(self, fake_pattern: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if not self.is_ready:
            return torch.zeros((), device=fake_pattern.device, dtype=fake_pattern.dtype)

        b, c, h, w = fake_pattern.shape
        if c != 3:
            return torch.zeros((), device=fake_pattern.device, dtype=fake_pattern.dtype)

        fake = fake_pattern.permute(0, 2, 3, 1).reshape(-1, 3)

        if mask is not None:
            m = mask.reshape(-1) > 0.5
            if int(m.sum().item()) > 0:
                fake = fake[m]

        if fake.numel() == 0:
            return torch.zeros((), device=fake_pattern.device, dtype=fake_pattern.dtype)

        palette = self.palette.to(device=fake_pattern.device, dtype=fake_pattern.dtype)  # [K,3]
        dists = torch.cdist(fake.unsqueeze(0), palette.unsqueeze(0)).squeeze(0)  # [N,K]
        min_dist = torch.min(dists, dim=1).values
        return torch.mean(min_dist)
