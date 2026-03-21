"""70x70 PatchGAN discriminator for HCAS-GAN."""

from __future__ import annotations

import torch
from torch import nn


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator that outputs a realism logit map.

    Input shape:
        [N, in_channels, H, W]
    Output shape:
        [N, 1, H_patch, W_patch]
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        c = int(base_channels)

        def block(in_c: int, out_c: int, *, stride: int, normalize: bool = True) -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1, bias=not normalize)
            ]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            block(in_channels, c, stride=2, normalize=False),
            block(c, c * 2, stride=2),
            block(c * 2, c * 4, stride=2),
            block(c * 4, c * 8, stride=1),
            nn.Conv2d(c * 8, 1, kernel_size=4, stride=1, padding=1),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
