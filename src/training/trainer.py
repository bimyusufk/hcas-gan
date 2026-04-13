"""HCAS-GAN trainer loops.

Implements point-4 core training workflow:
- forward/backward passes for Discriminator and Generator
- composite image construction using target mask
- saliency-aware Generator objective
- epoch-level metric aggregation for train/validation
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .loss import HCASLoss


@dataclass
class EpochMetrics:
    """Aggregated epoch metrics."""

    g_total: float
    g_adv: float
    g_sal: float
    g_lpips: float
    g_style: float
    g_palette: float
    g_freq: float
    d_total: float
    d_real: float
    d_fake: float
    num_batches: int
    num_samples: int
    duration_sec: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def _extract_batch(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if "image" not in batch or "mask" not in batch:
        raise KeyError("Batch dictionary must contain keys: 'image' and 'mask'.")

    image = batch["image"]
    mask = batch["mask"]

    if not isinstance(image, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError("Batch 'image' and 'mask' values must be torch.Tensor.")

    image = image.to(device=device, dtype=torch.float32, non_blocking=True)
    mask = mask.to(device=device, dtype=torch.float32, non_blocking=True)
    return image, mask


def _compose_with_mask(
    background: torch.Tensor,
    generated_pattern: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Blend generated pattern onto background inside mask area."""
    if generated_pattern.shape != background.shape:
        raise ValueError(
            "Generator output and background image must share identical shape. "
            f"Got {generated_pattern.shape} vs {background.shape}."
        )

    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"Mask must have shape [N,1,H,W], got: {mask.shape}")

    if mask.shape[0] != background.shape[0] or mask.shape[2:] != background.shape[2:]:
        raise ValueError(
            "Mask spatial shape must match background image. "
            f"Got mask={mask.shape}, image={background.shape}."
        )

    return (background * (1.0 - mask)) + (generated_pattern * mask)


def _resize_saliency_to_mask(saliency_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if saliency_map.ndim != 4:
        raise ValueError(f"Saliency output must be [N,C,H,W], got: {saliency_map.shape}")

    if saliency_map.shape[1] != 1:
        saliency_map = torch.mean(saliency_map, dim=1, keepdim=True)

    if saliency_map.shape[2:] != mask.shape[2:]:
        saliency_map = F.interpolate(
            saliency_map,
            size=mask.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

    return saliency_map


def _safe_mean(total: float, count: int) -> float:
    return float(total / count) if count > 0 else 0.0


def train_one_epoch(
    *,
    generator: nn.Module,
    discriminator: nn.Module,
    saliency_model: nn.Module,
    criterion: HCASLoss,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    log_interval: int = 0,
    max_grad_norm_g: float | None = None,
    max_grad_norm_d: float | None = None,
) -> EpochMetrics:
    """Train Generator and Discriminator for one epoch."""
    generator.train()
    discriminator.train()
    saliency_model.eval()

    running: dict[str, float] = {
        "g_total": 0.0,
        "g_adv": 0.0,
        "g_sal": 0.0,
        "g_lpips": 0.0,
        "g_style": 0.0,
        "g_palette": 0.0,
        "g_freq": 0.0,
        "d_total": 0.0,
        "d_real": 0.0,
        "d_fake": 0.0,
    }
    num_batches = 0
    num_samples = 0

    t0 = perf_counter()

    for step, batch in enumerate(dataloader, start=1):
        real_image, mask = _extract_batch(batch, device=device)
        batch_size = int(real_image.shape[0])

        # --- Train Discriminator ---
        _set_requires_grad(discriminator, True)
        optimizer_d.zero_grad(set_to_none=True)

        with torch.no_grad():
            fake_pattern_d = generator(real_image)
            fake_composite_d = _compose_with_mask(real_image, fake_pattern_d, mask)

        pred_real = discriminator(real_image)
        pred_fake = discriminator(fake_composite_d.detach())

        d_total, d_real, d_fake = criterion.discriminator_loss(pred_real, pred_fake)
        d_total.backward()

        if max_grad_norm_d is not None:
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_grad_norm_d)

        optimizer_d.step()

        # --- Train Generator ---
        _set_requires_grad(discriminator, False)
        optimizer_g.zero_grad(set_to_none=True)

        fake_pattern_g = generator(real_image)
        fake_composite_g = _compose_with_mask(real_image, fake_pattern_g, mask)

        pred_fake_for_g = discriminator(fake_composite_g)
        saliency_map = saliency_model(fake_composite_g)
        saliency_map = _resize_saliency_to_mask(saliency_map, mask)

        g_total, g_adv, g_sal, g_lpips, g_style, g_palette, g_freq = criterion.generator_loss(
            discriminator_pred=pred_fake_for_g,
            saliency_map=saliency_map,
            mask=mask,
            fake_pattern=fake_pattern_g,
            background=real_image,
            fake_composite=fake_composite_g,
            update_running_stats=True,
        )
        g_total.backward()

        if max_grad_norm_g is not None:
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_grad_norm_g)

        optimizer_g.step()

        # Restore D grads for next iteration and aggregate metrics
        _set_requires_grad(discriminator, True)

        running["g_total"] += float(g_total.detach().item()) * batch_size
        running["g_adv"] += float(g_adv.detach().item()) * batch_size
        running["g_sal"] += float(g_sal.detach().item()) * batch_size
        running["g_lpips"] += float(g_lpips.detach().item()) * batch_size
        running["g_style"] += float(g_style.detach().item()) * batch_size
        running["g_palette"] += float(g_palette.detach().item()) * batch_size
        running["g_freq"] += float(g_freq.detach().item()) * batch_size
        running["d_total"] += float(d_total.detach().item()) * batch_size
        running["d_real"] += float(d_real.detach().item()) * batch_size
        running["d_fake"] += float(d_fake.detach().item()) * batch_size
        num_batches += 1
        num_samples += batch_size

        if log_interval > 0 and step % log_interval == 0:
            print(
                f"[train step {step}] "
                f"g_total={float(g_total.detach().item()):.4f} "
                f"g_adv={float(g_adv.detach().item()):.4f} "
                f"g_sal={float(g_sal.detach().item()):.4f} "
                f"g_lpips={float(g_lpips.detach().item()):.4f} "
                f"g_style={float(g_style.detach().item()):.4f} "
                f"g_palette={float(g_palette.detach().item()):.4f} "
                f"g_freq={float(g_freq.detach().item()):.4f} "
                f"d_total={float(d_total.detach().item()):.4f}"
            )

    duration = perf_counter() - t0
    return EpochMetrics(
        g_total=_safe_mean(running["g_total"], num_samples),
        g_adv=_safe_mean(running["g_adv"], num_samples),
        g_sal=_safe_mean(running["g_sal"], num_samples),
        g_lpips=_safe_mean(running["g_lpips"], num_samples),
        g_style=_safe_mean(running["g_style"], num_samples),
        g_palette=_safe_mean(running["g_palette"], num_samples),
        g_freq=_safe_mean(running["g_freq"], num_samples),
        d_total=_safe_mean(running["d_total"], num_samples),
        d_real=_safe_mean(running["d_real"], num_samples),
        d_fake=_safe_mean(running["d_fake"], num_samples),
        num_batches=num_batches,
        num_samples=num_samples,
        duration_sec=float(duration),
    )


@torch.no_grad()
def validate_one_epoch(
    *,
    generator: nn.Module,
    discriminator: nn.Module,
    saliency_model: nn.Module,
    criterion: HCASLoss,
    dataloader: DataLoader,
    device: torch.device,
) -> EpochMetrics:
    """Evaluate current model parameters on a validation dataloader."""
    generator.eval()
    discriminator.eval()
    saliency_model.eval()

    running: dict[str, float] = {
        "g_total": 0.0,
        "g_adv": 0.0,
        "g_sal": 0.0,
        "g_lpips": 0.0,
        "g_style": 0.0,
        "g_palette": 0.0,
        "g_freq": 0.0,
        "d_total": 0.0,
        "d_real": 0.0,
        "d_fake": 0.0,
    }
    num_batches = 0
    num_samples = 0
    t0 = perf_counter()

    for batch in dataloader:
        real_image, mask = _extract_batch(batch, device=device)
        batch_size = int(real_image.shape[0])

        fake_pattern = generator(real_image)
        fake_composite = _compose_with_mask(real_image, fake_pattern, mask)

        pred_real = discriminator(real_image)
        pred_fake = discriminator(fake_composite)
        saliency_map = saliency_model(fake_composite)
        saliency_map = _resize_saliency_to_mask(saliency_map, mask)

        d_total, d_real, d_fake = criterion.discriminator_loss(pred_real, pred_fake)
        g_total, g_adv, g_sal, g_lpips, g_style, g_palette, g_freq = criterion.generator_loss(
            discriminator_pred=pred_fake,
            saliency_map=saliency_map,
            mask=mask,
            fake_pattern=fake_pattern,
            background=real_image,
            fake_composite=fake_composite,
            update_running_stats=False,
        )

        running["g_total"] += float(g_total.item()) * batch_size
        running["g_adv"] += float(g_adv.item()) * batch_size
        running["g_sal"] += float(g_sal.item()) * batch_size
        running["g_lpips"] += float(g_lpips.item()) * batch_size
        running["g_style"] += float(g_style.item()) * batch_size
        running["g_palette"] += float(g_palette.item()) * batch_size
        running["g_freq"] += float(g_freq.item()) * batch_size
        running["d_total"] += float(d_total.item()) * batch_size
        running["d_real"] += float(d_real.item()) * batch_size
        running["d_fake"] += float(d_fake.item()) * batch_size
        num_batches += 1
        num_samples += batch_size

    duration = perf_counter() - t0
    return EpochMetrics(
        g_total=_safe_mean(running["g_total"], num_samples),
        g_adv=_safe_mean(running["g_adv"], num_samples),
        g_sal=_safe_mean(running["g_sal"], num_samples),
        g_lpips=_safe_mean(running["g_lpips"], num_samples),
        g_style=_safe_mean(running["g_style"], num_samples),
        g_palette=_safe_mean(running["g_palette"], num_samples),
        g_freq=_safe_mean(running["g_freq"], num_samples),
        d_total=_safe_mean(running["d_total"], num_samples),
        d_real=_safe_mean(running["d_real"], num_samples),
        d_fake=_safe_mean(running["d_fake"], num_samples),
        num_batches=num_batches,
        num_samples=num_samples,
        duration_sec=float(duration),
    )
