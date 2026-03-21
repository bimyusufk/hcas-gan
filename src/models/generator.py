"""U-Net 256 style generator for HCAS-GAN.

Architecture notes:
- Encoder-decoder with skip connections (pix2pix-style U-Net)
- 8 downsampling stages for 256x256 inputs
- Output activation defaults to Sigmoid to match dataset normalization [0, 1]
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class _DownBlock(nn.Module):
    """Encoder block: Conv(stride=2) + optional norm + LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int, *, normalize: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not normalize)
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpBlock(nn.Module):
    """Decoder block: ConvTranspose(stride=2) + norm + ReLU (+ optional dropout)."""

    def __init__(self, in_channels: int, out_channels: int, *, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GeneratorUNet(nn.Module):
    """Full U-Net 256 generator with skip connections.

    Input shape:
        [N, in_channels, H, W]
    Output shape:
        [N, out_channels, H, W]
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        output_activation: str = "sigmoid",
    ):
        super().__init__()
        c = int(base_channels)

        # Encoder (downsampling path)
        self.down1 = _DownBlock(in_channels, c, normalize=False)  # 256 -> 128
        self.down2 = _DownBlock(c, c * 2)  # 128 -> 64
        self.down3 = _DownBlock(c * 2, c * 4)  # 64 -> 32
        self.down4 = _DownBlock(c * 4, c * 8)  # 32 -> 16
        self.down5 = _DownBlock(c * 8, c * 8)  # 16 -> 8
        self.down6 = _DownBlock(c * 8, c * 8)  # 8 -> 4
        self.down7 = _DownBlock(c * 8, c * 8)  # 4 -> 2
        self.down8 = _DownBlock(c * 8, c * 8, normalize=False)  # 2 -> 1

        # Decoder (upsampling path)
        self.up1 = _UpBlock(c * 8, c * 8, dropout=0.5)  # 1 -> 2
        self.up2 = _UpBlock(c * 16, c * 8, dropout=0.5)  # 2 -> 4
        self.up3 = _UpBlock(c * 16, c * 8, dropout=0.5)  # 4 -> 8
        self.up4 = _UpBlock(c * 16, c * 8)  # 8 -> 16
        self.up5 = _UpBlock(c * 16, c * 4)  # 16 -> 32
        self.up6 = _UpBlock(c * 8, c * 2)  # 32 -> 64
        self.up7 = _UpBlock(c * 4, c)  # 64 -> 128

        final_layers: list[nn.Module] = [
            nn.ConvTranspose2d(c * 2, out_channels, kernel_size=4, stride=2, padding=1)
        ]
        activation = output_activation.strip().lower()
        if activation == "sigmoid":
            final_layers.append(nn.Sigmoid())
        elif activation == "tanh":
            final_layers.append(nn.Tanh())
        else:
            raise ValueError(
                f"Unsupported output_activation: {output_activation!r}. Use 'sigmoid' or 'tanh'."
            )
        self.final = nn.Sequential(*final_layers)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias.data)

    @staticmethod
    def _concat_skip(decoder_feat: torch.Tensor, encoder_feat: torch.Tensor) -> torch.Tensor:
        if decoder_feat.shape[2:] != encoder_feat.shape[2:]:
            decoder_feat = F.interpolate(
                decoder_feat,
                size=encoder_feat.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        return torch.cat((decoder_feat, encoder_feat), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        u1 = self._concat_skip(u1, d7)

        u2 = self.up2(u1)
        u2 = self._concat_skip(u2, d6)

        u3 = self.up3(u2)
        u3 = self._concat_skip(u3, d5)

        u4 = self.up4(u3)
        u4 = self._concat_skip(u4, d4)

        u5 = self.up5(u4)
        u5 = self._concat_skip(u5, d3)

        u6 = self.up6(u5)
        u6 = self._concat_skip(u6, d2)

        u7 = self.up7(u6)
        u7 = self._concat_skip(u7, d1)

        out = self.final(u7)
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out
