"""Random-crop LPIPS training loss for HCAS-GAN.

This module keeps the LPIPS backbone frozen while preserving gradient flow
to the Generator output. It samples several random crops per sample and uses
the AlexNet LPIPS backbone by default.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import torch
import torch.nn.functional as F
from torch import nn


def _to_lpips_range(x: torch.Tensor) -> torch.Tensor:
    """Convert tensors from [0,1] to [-1,1] as expected by LPIPS."""
    if x.ndim != 4:
        raise ValueError(f"Expected tensor [N,C,H,W], got: {tuple(x.shape)}")
    if x.shape[1] not in (1, 3):
        raise ValueError(f"Channel dimension must be 1 or 3, got: {int(x.shape[1])}")

    y = x.float().clamp(0.0, 1.0)
    if y.shape[1] == 1:
        y = y.repeat(1, 3, 1, 1)
    return (y * 2.0) - 1.0


def _pad_sample_to_crop(sample: torch.Tensor, crop_size: int, *, is_mask: bool) -> torch.Tensor:
    """Pad a single sample [C,H,W] (or [1,H,W] for masks) to fit a crop."""
    if sample.ndim != 3:
        raise ValueError(f"Expected 3D sample tensor, got: {tuple(sample.shape)}")

    _, h, w = sample.shape
    pad_h = max(int(crop_size) - h, 0)
    pad_w = max(int(crop_size) - w, 0)
    if pad_h == 0 and pad_w == 0:
        return sample

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if is_mask:
        return F.pad(sample, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)

    # Replicate padding is robust for very small images and keeps colours stable.
    return F.pad(sample, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")


@dataclass(frozen=True)
class _CropBox:
    top: int
    left: int


class RandomCropLPIPSLoss(nn.Module):
    """LPIPS loss over multiple random crops.

    The LPIPS module is kept in eval mode with frozen parameters, but gradients
    are allowed to flow to the image inputs so the Generator can be optimized.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        crop_size: int = 128,
        num_crops: int = 2,
        net: str = "alex",
        min_mask_coverage: float = 0.0,
        max_resample_attempts: int = 8,
    ):
        super().__init__()
        self.requested_enabled = bool(enabled)
        self.crop_size = max(1, int(crop_size))
        self.num_crops = max(1, int(num_crops))
        self.net = str(net)
        self.min_mask_coverage = max(0.0, float(min_mask_coverage))
        self.max_resample_attempts = max(1, int(max_resample_attempts))
        self.initialization_error: str | None = None

        self.lpips_model: nn.Module | None = None
        if self.requested_enabled:
            try:
                import lpips  # type: ignore

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    model = lpips.LPIPS(net=self.net)
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False
                self.lpips_model = model
            except Exception as exc:  # pragma: no cover - depends on optional dependency availability
                self.initialization_error = f"{type(exc).__name__}: {exc}"

    @property
    def is_ready(self) -> bool:
        return bool(self.requested_enabled and self.lpips_model is not None)

    @property
    def backend(self) -> str:
        return f"lpips:{self.net}" if self.is_ready else "unavailable"

    def _ensure_model_device(self, device: torch.device) -> None:
        if self.lpips_model is None:
            return

        first_param = next(self.lpips_model.parameters(), None)
        if first_param is not None and first_param.device != device:
            self.lpips_model = self.lpips_model.to(device)

    def _sample_crop_box(
        self,
        *,
        mask_2d: torch.Tensor | None,
        foreground_coords: torch.Tensor | None,
        height: int,
        width: int,
    ) -> _CropBox:
        crop_size = self.crop_size
        device = mask_2d.device if mask_2d is not None else (
            foreground_coords.device if foreground_coords is not None else torch.device("cpu")
        )

        top = 0
        left = 0
        for _ in range(self.max_resample_attempts):
            if foreground_coords is not None and foreground_coords.numel() > 0:
                idx = int(torch.randint(0, foreground_coords.shape[0], (1,), device=device).item())
                center_y = int(foreground_coords[idx, 0].item())
                center_x = int(foreground_coords[idx, 1].item())
                top = max(0, min(center_y - crop_size // 2, height - crop_size))
                left = max(0, min(center_x - crop_size // 2, width - crop_size))
            else:
                top = int(torch.randint(0, max(height - crop_size + 1, 1), (1,), device=device).item())
                left = int(torch.randint(0, max(width - crop_size + 1, 1), (1,), device=device).item())

            if mask_2d is None or self.min_mask_coverage <= 0.0:
                return _CropBox(top=top, left=left)

            crop_mask = mask_2d[top : top + crop_size, left : left + crop_size]
            coverage = float(crop_mask.mean().item()) if crop_mask.numel() > 0 else 0.0
            if coverage >= self.min_mask_coverage:
                return _CropBox(top=top, left=left)

        return _CropBox(top=top, left=left)

    def _sample_sample_loss(
        self,
        *,
        fake_sample: torch.Tensor,
        reference_sample: torch.Tensor,
        mask_sample: torch.Tensor | None,
    ) -> torch.Tensor:
        fake_sample = _pad_sample_to_crop(fake_sample, self.crop_size, is_mask=False)
        reference_sample = _pad_sample_to_crop(reference_sample, self.crop_size, is_mask=False)

        mask_2d: torch.Tensor | None = None
        foreground_coords: torch.Tensor | None = None
        if mask_sample is not None:
            mask_sample = _pad_sample_to_crop(mask_sample, self.crop_size, is_mask=True)
            mask_2d = mask_sample.squeeze(0)
            foreground_coords = torch.nonzero(mask_2d > 0.5, as_tuple=False)

        height = int(fake_sample.shape[-2])
        width = int(fake_sample.shape[-1])

        fake_crops: list[torch.Tensor] = []
        ref_crops: list[torch.Tensor] = []
        for _ in range(self.num_crops):
            crop_box = self._sample_crop_box(
                mask_2d=mask_2d,
                foreground_coords=foreground_coords,
                height=height,
                width=width,
            )
            top, left = crop_box.top, crop_box.left
            fake_crops.append(fake_sample[:, top : top + self.crop_size, left : left + self.crop_size])
            ref_crops.append(reference_sample[:, top : top + self.crop_size, left : left + self.crop_size])

        fake_batch = torch.stack(fake_crops, dim=0)
        ref_batch = torch.stack(ref_crops, dim=0)

        fake_lpips = _to_lpips_range(fake_batch)
        ref_lpips = _to_lpips_range(ref_batch)
        scores = self.lpips_model(fake_lpips, ref_lpips)
        return scores.reshape(-1).mean()

    def forward(
        self,
        fake_composite: torch.Tensor,
        reference_image: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.is_ready:
            return torch.zeros((), device=fake_composite.device, dtype=fake_composite.dtype)

        if fake_composite.shape != reference_image.shape:
            raise ValueError(
                "fake_composite and reference_image must have identical shapes. "
                f"Got {tuple(fake_composite.shape)} vs {tuple(reference_image.shape)}."
            )
        if fake_composite.ndim != 4:
            raise ValueError(f"Expected tensors [N,C,H,W], got: {tuple(fake_composite.shape)}")
        if fake_composite.shape[0] == 0:
            raise ValueError("Empty batch is not allowed.")

        if mask is not None:
            if mask.ndim != 4 or mask.shape[1] != 1:
                raise ValueError(f"Mask must have shape [N,1,H,W], got: {tuple(mask.shape)}")
            if mask.shape[0] != fake_composite.shape[0] or mask.shape[2:] != fake_composite.shape[2:]:
                raise ValueError(
                    "Mask shape must match the image batch spatial dimensions. "
                    f"Got mask={tuple(mask.shape)} image={tuple(fake_composite.shape)}."
                )

        self._ensure_model_device(fake_composite.device)

        total = torch.zeros((), device=fake_composite.device, dtype=fake_composite.dtype)
        batch_size = int(fake_composite.shape[0])

        for idx in range(batch_size):
            mask_sample = mask[idx] if mask is not None else None
            sample_loss = self._sample_sample_loss(
                fake_sample=fake_composite[idx],
                reference_sample=reference_image[idx],
                mask_sample=mask_sample,
            )
            total = total + sample_loss

        return total / float(batch_size)
