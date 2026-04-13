"""Smoke test for HCAS-GAN v3.

Verifies that the v3 configuration can:
- load successfully
- build Generator / Discriminator / saliency / LPIPS losses
- run one forward/backward training step
- persist and restore EMA loss-balancing state via checkpoint
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch
import yaml
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.deepgaze_wrapper import DeepGazeWrapper
from src.models.discriminator import PatchDiscriminator
from src.models.generator import GeneratorUNet
from src.training.checkpoint import CheckpointManager
from src.training.loss import HCASLoss
from src.training.loss_lpips import RandomCropLPIPSLoss
from src.utils.image_size import coerce_image_size_hw, format_image_size_wh


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for HCAS-GAN v3")
    parser.add_argument(
        "--config",
        type=str,
        default="config.run_v3.yaml",
        help="Path to the v3 config YAML",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, or cuda",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Synthetic batch size for the smoke test",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Synthetic image size (defaults to config.hcas_specific.image_size)",
    )
    parser.add_argument(
        "--saliency-backend",
        type=str,
        default="proxy",
        help="DeepGaze backend for the smoke test (proxy is safest offline)",
    )
    parser.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help="Skip checkpoint save/load validation",
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    mode = str(value).strip().lower()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_synthetic_batch(
    batch_size: int,
    image_size_hw: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = image_size_hw
    background = torch.rand(batch_size, 3, height, width, device=device)
    mask = torch.zeros(batch_size, 1, height, width, device=device)

    target_h = max(32, height // 4)
    target_w = max(32, width // 4)
    top = max(0, (height - target_h) // 2)
    left = max(0, (width - target_w) // 2)
    mask[:, :, top : top + target_h, left : left + target_w] = 1.0
    return background, mask


def grad_norm(module: nn.Module) -> float:
    total = 0.0
    for param in module.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().abs().sum().item())
    return total


def main() -> None:
    args = parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = load_config(config_path)

    seed = int(config.get("experiment", {}).get("seed", 42))
    set_seed(seed)

    device = resolve_device(args.device)
    image_size_hw = coerce_image_size_hw(args.image_size or config.get("hcas_specific", {}).get("image_size", 256))
    batch_size = int(args.batch_size)

    lpips_cfg = config.get("lpips", {})
    balance_cfg = config.get("loss_balance", {})
    saliency_cfg = config.get("saliency_model", {})

    print("[SMOKE] ===== HCAS-GAN v3 smoke test =====")
    print(f"[SMOKE] config          : {config_path}")
    print(f"[SMOKE] device          : {device}")
    print(f"[SMOKE] batch / image   : {batch_size} / {format_image_size_wh(image_size_hw)}")
    print(f"[SMOKE] saliency backend: {args.saliency_backend}")
    print(f"[SMOKE] lpips backend   : {lpips_cfg.get('net', 'alex')}")
    print(f"[SMOKE] balance mode    : {'ema' if bool(balance_cfg.get('enabled', False)) else 'off'}")

    generator = GeneratorUNet().to(device)
    discriminator = PatchDiscriminator().to(device)
    saliency_model = DeepGazeWrapper(
        backend=args.saliency_backend,
        model_name=str(saliency_cfg.get("model_name", "DeepGazeIII")),
        pretrained=bool(saliency_cfg.get("pretrained", True)),
        trust_repo=bool(saliency_cfg.get("trust_repo", True)),
        repo=str(saliency_cfg.get("repo", "matthias-k/DeepGaze")),
        centerbias_mode=str(saliency_cfg.get("centerbias_mode", "zeros")),
        fixation_history_length=int(saliency_cfg.get("fixation_history_length", 4)),
        output_mode=str(saliency_cfg.get("output_mode", "density")),
        input_range=str(saliency_cfg.get("input_range", "0_1")),
        allow_fallback=bool(saliency_cfg.get("allow_fallback", True)),
        verbose=False,
    ).to(device)

    lpips_loss = RandomCropLPIPSLoss(
        enabled=True,
        crop_size=int(lpips_cfg.get("crop_size", 128)),
        num_crops=int(lpips_cfg.get("num_crops", 1)),
        net=str(lpips_cfg.get("net", "alex")),
        min_mask_coverage=float(lpips_cfg.get("min_mask_coverage", 0.0)),
        max_resample_attempts=int(lpips_cfg.get("max_resample_attempts", 8)),
    )
    if not lpips_loss.is_ready:
        raise RuntimeError(f"LPIPS could not be initialized: {lpips_loss.initialization_error or 'unknown reason'}")

    criterion = HCASLoss(
        lambda_sal=float(config.get("hcas_specific", {}).get("lambda_sal", 1.0)),
        lambda_lpips=float(lpips_cfg.get("lambda_lpips", 1.0)),
        normalize_saliency=bool(balance_cfg.get("enabled", False)) and bool(balance_cfg.get("normalize_saliency", True)),
        normalize_lpips=bool(balance_cfg.get("enabled", False)) and bool(balance_cfg.get("normalize_lpips", True)),
        ema_momentum=float(balance_cfg.get("ema_momentum", 0.99)),
        ema_eps=float(balance_cfg.get("ema_eps", 1e-8)),
        lpips_loss=lpips_loss,
    ).to(device)

    background, mask = build_synthetic_batch(batch_size=batch_size, image_size_hw=image_size_hw, device=device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # --- discriminator step ---
    generator.train()
    discriminator.train()
    saliency_model.eval()

    optimizer_d.zero_grad(set_to_none=True)
    with torch.no_grad():
        fake_pattern_d = generator(background)
        fake_composite_d = background * (1.0 - mask) + fake_pattern_d * mask
    pred_real = discriminator(background)
    pred_fake = discriminator(fake_composite_d.detach())
    d_total, d_real, d_fake = criterion.discriminator_loss(pred_real, pred_fake)
    d_total.backward()
    optimizer_d.step()

    # --- generator step ---
    optimizer_g.zero_grad(set_to_none=True)
    fake_pattern_g = generator(background)
    fake_composite_g = background * (1.0 - mask) + fake_pattern_g * mask
    pred_fake_g = discriminator(fake_composite_g)
    saliency_map = saliency_model(fake_composite_g)

    g_total, g_adv, g_sal, g_lpips, *_ = criterion.generator_loss(
        discriminator_pred=pred_fake_g,
        saliency_map=saliency_map,
        mask=mask,
        fake_pattern=fake_pattern_g,
        background=background,
        fake_composite=fake_composite_g,
        update_running_stats=True,
    )
    g_total.backward()
    g_grad = grad_norm(generator)
    optimizer_g.step()

    print(
        "[SMOKE] losses         : "
        f"g_total={float(g_total.item()):.6f} | g_adv={float(g_adv.item()):.6f} | "
        f"g_sal={float(g_sal.item()):.6f} | g_lpips={float(g_lpips.item()):.6f} | "
        f"d_total={float(d_total.item()):.6f}"
    )
    print(f"[SMOKE] generator grad : {g_grad:.6f}")
    print(
        "[SMOKE] EMA state       : "
        f"sal={criterion.saliency_loss_ema_value:.6f} | lpips={criterion.lpips_loss_ema_value:.6f}"
    )

    assert g_grad > 0.0, "Generator gradient should be non-zero."
    assert torch.isfinite(g_total), "Generator loss must be finite."
    assert torch.isfinite(d_total), "Discriminator loss must be finite."

    if not args.skip_checkpoint:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, keep_last_n=2)
            ckpt_path = manager.save(
                epoch=1,
                step=1,
                generator=generator,
                discriminator=discriminator,
                criterion=criterion,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
            )
            loaded_criterion = HCASLoss(
                lambda_sal=float(config.get("hcas_specific", {}).get("lambda_sal", 1.0)),
                lambda_lpips=float(lpips_cfg.get("lambda_lpips", 1.0)),
                normalize_saliency=bool(balance_cfg.get("enabled", False)) and bool(balance_cfg.get("normalize_saliency", True)),
                normalize_lpips=bool(balance_cfg.get("enabled", False)) and bool(balance_cfg.get("normalize_lpips", True)),
                ema_momentum=float(balance_cfg.get("ema_momentum", 0.99)),
                ema_eps=float(balance_cfg.get("ema_eps", 1e-8)),
                lpips_loss=RandomCropLPIPSLoss(
                    enabled=True,
                    crop_size=int(lpips_cfg.get("crop_size", 128)),
                    num_crops=int(lpips_cfg.get("num_crops", 1)),
                    net=str(lpips_cfg.get("net", "alex")),
                    min_mask_coverage=float(lpips_cfg.get("min_mask_coverage", 0.0)),
                    max_resample_attempts=int(lpips_cfg.get("max_resample_attempts", 8)),
                ),
            ).to(device)
            manager.load(
                ckpt_path,
                generator=generator,
                discriminator=discriminator,
                criterion=loaded_criterion,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                device=device,
            )
            print(
                "[SMOKE] checkpoint     : "
                f"ok | sal={loaded_criterion.saliency_loss_ema_value:.6f} | lpips={loaded_criterion.lpips_loss_ema_value:.6f}"
            )

    print("[SMOKE] SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
