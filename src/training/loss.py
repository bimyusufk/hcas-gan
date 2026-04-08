"""HCAS-GAN loss functions.

Implements:
- Generator objective: L_G = L_adv + lambda_sal * L_sal
- Discriminator BCE loss for real/fake logits
"""

from __future__ import annotations

import torch
from torch import nn

from .loss_frequency import FrequencyLoss
from .loss_palette import PaletteLoss
from .loss_style import StylePriorLoss


class HCASLoss(nn.Module):
    """Composite loss for HCAS-GAN training."""

    def __init__(
        self,
        lambda_sal: float = 15.0,
        lambda_style: float = 0.0,
        lambda_palette: float = 0.0,
        lambda_freq: float = 0.0,
        style_loss: StylePriorLoss | None = None,
        palette_loss: PaletteLoss | None = None,
        frequency_loss: FrequencyLoss | None = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.lambda_sal = float(lambda_sal)
        self.lambda_style = float(lambda_style)
        self.lambda_palette = float(lambda_palette)
        self.lambda_freq = float(lambda_freq)
        self.eps = float(eps)
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.style_loss = style_loss
        self.palette_loss = palette_loss
        self.frequency_loss = frequency_loss

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

    def generator_loss(
        self,
        discriminator_pred: torch.Tensor,
        saliency_map: torch.Tensor,
        mask: torch.Tensor,
        fake_pattern: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute generator objective.

        L_G = L_adv + lambda_sal * L_sal
        """
        valid_labels = torch.ones_like(discriminator_pred)
        loss_adv = self.adversarial_loss(discriminator_pred, valid_labels)

        loss_sal, _ = self.compute_saliency_penalty(saliency_map, mask)

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

        total_g_loss = (
            loss_adv
            + (self.lambda_sal * loss_sal)
            + (self.lambda_style * loss_style)
            + (self.lambda_palette * loss_palette)
            + (self.lambda_freq * loss_freq)
        )
        return total_g_loss, loss_adv, loss_sal, loss_style, loss_palette, loss_freq

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
