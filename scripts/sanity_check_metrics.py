"""Sanity check for SSIM/LPIPS utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    from src.utils import compute_lpips_batch, compute_ssim_batch

    torch.manual_seed(42)

    x = torch.rand(4, 3, 64, 64)
    y = x.clone()
    y[0] = torch.clamp(y[0] * 0.9 + 0.05, 0.0, 1.0)

    mean_ssim, ssim_vals = compute_ssim_batch(x, y)
    print("ssim_mean", round(float(mean_ssim), 6))
    print("ssim_first_two", [round(float(v), 6) for v in ssim_vals[:2]])

    lpips_res = compute_lpips_batch(x, y, net="alex")
    print("lpips_available", lpips_res.available)
    print("lpips_backend", lpips_res.backend)
    print("lpips_mean", round(float(lpips_res.mean), 6))

    # Basic expectations:
    # - identical-ish pairs should still produce valid metric values.
    # - SSIM should be in [-1, 1], usually near 1 for near-identical images.
    if not (-1.0 <= mean_ssim <= 1.0):
        raise RuntimeError(f"SSIM out of range: {mean_ssim}")

    print("[OK] metrics sanity passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
