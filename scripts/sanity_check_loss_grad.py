"""Sanity check for HCASLoss gradient flow.

Validates that:
1) Generator parameters receive non-zero gradients.
2) Saliency branch contributes gradients to Generator.
3) Frozen saliency-model parameters do not receive gradients.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.loss import HCASLoss


class ToyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, background: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([background, mask], dim=1)
        return self.net(x)


class ToyDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class FrozenSaliencyProxy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid(),
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


def grad_norm(module: nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().abs().sum().item())
    return total


def main() -> None:
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, h, w = 2, 64, 64

    generator = ToyGenerator().to(device)
    discriminator = ToyDiscriminator().to(device)
    saliency = FrozenSaliencyProxy().to(device)
    criterion = HCASLoss(lambda_sal=15.0)

    background = torch.rand(batch_size, 3, h, w, device=device)
    mask = torch.zeros(batch_size, 1, h, w, device=device)
    mask[:, :, 16:48, 16:48] = 1.0

    # Check total generator objective gradient path
    generator.zero_grad(set_to_none=True)
    fake_pattern = generator(background, mask)
    composite = background * (1.0 - mask) + fake_pattern * mask
    disc_pred = discriminator(composite)
    saliency_map = saliency(composite)

    total_g, loss_adv, loss_sal = criterion.generator_loss(disc_pred, saliency_map, mask)
    total_g.backward()
    g_grad_total = grad_norm(generator)

    # Check saliency-only path gradient (critical for HCAS behavior)
    generator.zero_grad(set_to_none=True)
    fake_pattern2 = generator(background, mask)
    composite2 = background * (1.0 - mask) + fake_pattern2 * mask
    saliency_map2 = saliency(composite2)
    sal_only, _ = criterion.compute_saliency_penalty(saliency_map2, mask)
    sal_only.backward()
    g_grad_sal = grad_norm(generator)

    saliency_has_grad = any(p.grad is not None for p in saliency.parameters())

    print(f"device={device}")
    print(f"loss_adv={float(loss_adv.item()):.6f}")
    print(f"loss_sal={float(loss_sal.item()):.6f}")
    print(f"total_g={float(total_g.item()):.6f}")
    print(f"generator_grad_total={g_grad_total:.6f}")
    print(f"generator_grad_saliency_only={g_grad_sal:.6f}")
    print(f"frozen_saliency_params_have_grad={saliency_has_grad}")

    assert g_grad_total > 0.0, "Generator grad is zero for total loss."
    assert g_grad_sal > 0.0, "Generator grad is zero for saliency-only loss."
    assert not saliency_has_grad, "Frozen saliency params unexpectedly received gradients."

    print("SANITY CHECK PASSED")


if __name__ == "__main__":
    main()
