"""Sanity check for DeepGazeWrapper integration.

Validates:
1) Wrapper initializes using configured backend (real or fallback).
2) Forward pass output shape is [N,1,H,W] and finite.
3) Gradient flows from saliency output back to image input.
4) Fallback path works when real backend initialization fails.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.deepgaze_wrapper import DeepGazeWrapper


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(config: dict) -> torch.device:
    preferred = str(config.get("hardware", {}).get("device", "cpu")).lower()
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    _set_seed(seed)

    saliency_cfg = cfg.get("saliency_model", {})
    device = _resolve_device(cfg)

    model = DeepGazeWrapper(
        backend=str(saliency_cfg.get("backend", "auto")),
        model_name=str(saliency_cfg.get("model_name", "DeepGazeIII")),
        pretrained=bool(saliency_cfg.get("pretrained", True)),
        trust_repo=bool(saliency_cfg.get("trust_repo", True)),
        repo=str(saliency_cfg.get("repo", "matthias-k/DeepGaze")),
        centerbias_mode=str(saliency_cfg.get("centerbias_mode", "zeros")),
        fixation_history_length=int(saliency_cfg.get("fixation_history_length", 4)),
        output_mode=str(saliency_cfg.get("output_mode", "density")),
        input_range=str(saliency_cfg.get("input_range", "0_1")),
        allow_fallback=bool(saliency_cfg.get("allow_fallback", True)),
        verbose=bool(saliency_cfg.get("verbose", True)),
    ).to(device)

    image = torch.rand(2, 3, 64, 64, device=device, requires_grad=True)
    saliency = model(image)

    if saliency.ndim != 4 or saliency.shape[1] != 1 or saliency.shape[2:] != image.shape[2:]:
        raise AssertionError(f"Unexpected saliency shape: {tuple(saliency.shape)}")
    if not torch.isfinite(saliency).all():
        raise AssertionError("Saliency output contains non-finite values")

    loss = saliency.mean()
    loss.backward()

    if image.grad is None:
        raise AssertionError("Input image gradient is missing")

    grad_norm = float(image.grad.detach().abs().sum().item())
    if grad_norm <= 0.0:
        raise AssertionError("Input image gradient norm is zero")

    info = model.get_backend_info()

    # Force fallback path by using invalid repo while fallback is enabled.
    forced_fallback_model = DeepGazeWrapper(
        backend="deepgazeiii_torchhub",
        model_name="DeepGazeIII",
        pretrained=False,
        trust_repo=True,
        repo="invalid/repo-does-not-exist",
        allow_fallback=True,
        verbose=False,
    ).to(device)
    forced_info = forced_fallback_model.get_backend_info()

    if forced_info.active_backend != "proxy":
        raise AssertionError("Forced fallback test failed: expected proxy backend")

    print(f"device={device}")
    print(
        "backend_info="
        + json.dumps(
            {
                "requested": info.requested_backend,
                "active": info.active_backend,
                "model": info.model_name,
                "using_fallback": info.using_fallback,
                "fallback_reason": info.fallback_reason,
            },
            indent=2,
        )
    )
    print(
        "forced_fallback_info="
        + json.dumps(
            {
                "requested": forced_info.requested_backend,
                "active": forced_info.active_backend,
                "using_fallback": forced_info.using_fallback,
                "fallback_reason": forced_info.fallback_reason,
            },
            indent=2,
        )
    )
    print(f"input_grad_norm={grad_norm:.6f}")
    print("DEEPGAZE WRAPPER SANITY CHECK PASSED")


if __name__ == "__main__":
    main()
