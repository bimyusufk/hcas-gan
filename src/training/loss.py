"""HCAS-GAN loss functions.

Implements:
- Generator objective: L_G = L_adv + lambda_sal * L_sal + lambda_lpips * L_lpips
- EMA / running-average normalization for saliency and LPIPS loss scales
- Optional style / palette / frequency priors for compatibility with older runs
- Discriminator BCE loss for real/fake logits
"""

from __future__ import annotations

import torch
from torch import nn

from .loss_frequency import FrequencyLoss
from .loss_lpips import RandomCropLPIPSLoss
from .loss_palette import PaletteLoss
from .loss_style import StylePriorLoss


class HCASLoss(nn.Module):
    """Composite loss for HCAS-GAN training."""

    def __init__(
        self,
        lambda_sal: float = 15.0,
        lambda_lpips: float = 0.0,
        lambda_style: float = 0.0,
        lambda_palette: float = 0.0,
        lambda_freq: float = 0.0,
        normalize_saliency: bool = False,
        normalize_lpips: bool = False,
        ema_momentum: float = 0.99,
        ema_eps: float = 1e-8,
        lpips_loss: RandomCropLPIPSLoss | None = None,
        style_loss: StylePriorLoss | None = None,
        palette_loss: PaletteLoss | None = None,
        frequency_loss: FrequencyLoss | None = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.lambda_sal = float(lambda_sal)
        self.lambda_lpips = float(lambda_lpips)
        self.lambda_style = float(lambda_style)
        self.lambda_palette = float(lambda_palette)
        self.lambda_freq = float(lambda_freq)
        self.normalize_saliency = bool(normalize_saliency)
        self.normalize_lpips = bool(normalize_lpips)
        self.ema_momentum = float(ema_momentum)
        self.ema_eps = float(ema_eps)
        self.eps = float(eps)
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.lpips_loss = lpips_loss
        self.style_loss = style_loss
        self.palette_loss = palette_loss
        self.frequency_loss = frequency_loss

        if not 0.0 <= self.ema_momentum < 1.0:
            raise ValueError(f"ema_momentum must be in [0, 1), got {self.ema_momentum}")
        if self.ema_eps <= 0.0:
            raise ValueError(f"ema_eps must be > 0, got {self.ema_eps}")

        self.register_buffer("saliency_loss_ema", torch.zeros((), dtype=torch.float32), persistent=True)
        self.register_buffer("lpips_loss_ema", torch.zeros((), dtype=torch.float32), persistent=True)
        self.register_buffer(
            "saliency_loss_ema_initialized",
            torch.zeros((), dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "lpips_loss_ema_initialized",
            torch.zeros((), dtype=torch.bool),
            persistent=True,
        )

    def compute_saliency_penalty(
        self,
        saliency_map: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute normalized saliency-over-mask penalty.

        Returns:
            mean_penalty: scalar tensor for optimization
            per_sample_penalty: per-batch element values
        """
        if saliency_map.shape != mask.shape:
            raise ValueError(
                "saliency_map and mask must have identical shapes. "
                f"Got {saliency_map.shape} vs {mask.shape}."
            )

        mask = mask.float()
        attention_on_target = saliency_map * mask

        target_area_sum = torch.sum(mask, dim=(1, 2, 3)).clamp_min(self.eps)
        total_attention = torch.sum(attention_on_target, dim=(1, 2, 3))

        per_sample_penalty = total_attention / target_area_sum
        mean_penalty = torch.mean(per_sample_penalty)
        return mean_penalty, per_sample_penalty

    def _update_running_average(
        self,
        *,
        loss_value: torch.Tensor,
        ema_buffer: torch.Tensor,
        initialized_buffer: torch.Tensor,
        update_running_stats: bool,
    ) -> torch.Tensor:
        """Update an EMA buffer and return the denominator used for normalization."""
        current = loss_value.detach().to(device=ema_buffer.device, dtype=ema_buffer.dtype)

        if update_running_stats:
            if bool(initialized_buffer.item()):
                ema_buffer.mul_(self.ema_momentum).add_(current * (1.0 - self.ema_momentum))
            else:
                ema_buffer.copy_(current)
                initialized_buffer.fill_(True)

        if bool(initialized_buffer.item()):
            return ema_buffer.clamp_min(self.ema_eps)
        return current.clamp_min(self.ema_eps)

    def _normalize_with_running_average(
        self,
        *,
        loss_value: torch.Tensor,
        ema_buffer: torch.Tensor,
        initialized_buffer: torch.Tensor,
        update_running_stats: bool,
    ) -> torch.Tensor:
        denom = self._update_running_average(
            loss_value=loss_value,
            ema_buffer=ema_buffer,
            initialized_buffer=initialized_buffer,
            update_running_stats=update_running_stats,
        )
        return loss_value / denom

    @property
    def saliency_loss_ema_value(self) -> float:
        return float(self.saliency_loss_ema.item())

    @property
    def lpips_loss_ema_value(self) -> float:
        return float(self.lpips_loss_ema.item())

    def generator_loss(
        self,
        discriminator_pred: torch.Tensor,
        saliency_map: torch.Tensor,
        mask: torch.Tensor,
        fake_pattern: torch.Tensor | None = None,
        background: torch.Tensor | None = None,
        fake_composite: torch.Tensor | None = None,
        update_running_stats: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Compute generator objective.

        L_G = L_adv + lambda_sal * L_sal + lambda_lpips * L_lpips
        """
        valid_labels = torch.ones_like(discriminator_pred)
        loss_adv = self.adversarial_loss(discriminator_pred, valid_labels)

        loss_sal_raw, _ = self.compute_saliency_penalty(saliency_map, mask)
        loss_sal = loss_sal_raw
        if self.normalize_saliency:
            loss_sal = self._normalize_with_running_average(
                loss_value=loss_sal_raw,
                ema_buffer=self.saliency_loss_ema,
                initialized_buffer=self.saliency_loss_ema_initialized,
                update_running_stats=update_running_stats,
            )

        loss_lpips_raw = torch.zeros((), device=discriminator_pred.device, dtype=discriminator_pred.dtype)
        loss_lpips = loss_lpips_raw

        zero = torch.zeros((), device=discriminator_pred.device, dtype=discriminator_pred.dtype)
        loss_style = zero
        loss_palette = zero
        loss_freq = zero

        if fake_pattern is not None:
            if self.style_loss is not None and self.lambda_style > 0.0:
                loss_style = self.style_loss(fake_pattern=fake_pattern, mask=mask)
            if self.palette_loss is not None and self.lambda_palette > 0.0:
                loss_palette = self.palette_loss(fake_pattern=fake_pattern, mask=mask)
            if self.frequency_loss is not None and self.lambda_freq > 0.0:
                loss_freq = self.frequency_loss(fake_pattern=fake_pattern, mask=mask)

        if (
            self.lpips_loss is not None
            and self.lambda_lpips > 0.0
        ):
            if fake_composite is None or background is None:
                raise ValueError(
                    "LPIPS loss requires both 'background' and 'fake_composite' tensors when enabled."
                )
            loss_lpips_raw = self.lpips_loss(
                fake_composite=fake_composite,
                reference_image=background,
                mask=mask,
            )
            loss_lpips = loss_lpips_raw
            if self.normalize_lpips:
                loss_lpips = self._normalize_with_running_average(
                    loss_value=loss_lpips_raw,
                    ema_buffer=self.lpips_loss_ema,
                    initialized_buffer=self.lpips_loss_ema_initialized,
                    update_running_stats=update_running_stats,
                )

        total_g_loss = (
            loss_adv
            + (self.lambda_sal * loss_sal)
            + (self.lambda_lpips * loss_lpips)
            + (self.lambda_style * loss_style)
            + (self.lambda_palette * loss_palette)
            + (self.lambda_freq * loss_freq)
        )
        return total_g_loss, loss_adv, loss_sal, loss_lpips, loss_style, loss_palette, loss_freq

    def discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Standard discriminator BCE objective for real/fake logits."""
        real_labels = torch.ones_like(real_pred)
        fake_labels = torch.zeros_like(fake_pred)

        loss_real = self.adversarial_loss(real_pred, real_labels)
        loss_fake = self.adversarial_loss(fake_pred, fake_labels)
        total_d_loss = 0.5 * (loss_real + loss_fake)
        return total_d_loss, loss_real, loss_fake
