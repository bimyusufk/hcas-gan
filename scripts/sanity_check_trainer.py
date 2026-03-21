"""Sanity check for HCAS-GAN trainer loop.

Runs a tiny train/validation cycle to verify:
1) forward/backward pass works end-to-end
2) optimizer updates execute without runtime errors
3) epoch metrics are finite and well-formed
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import DatasetConfig, LabelMeCamouflageDataset, build_augmentations, build_resize_collate_fn
from src.models.deepgaze_wrapper import DeepGazeWrapper
from src.models.discriminator import PatchDiscriminator
from src.models.generator import GeneratorUNet
from src.training.loss import HCASLoss
from src.training.trainer import train_one_epoch, validate_one_epoch


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
    config_path = PROJECT_ROOT / "config.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    seed = int(cfg.get("experiment", {}).get("seed", 42))
    _set_seed(seed)

    dataset_cfg = DatasetConfig(
        annotations_dir=str((PROJECT_ROOT / cfg["data"]["annotations_dir"]).resolve()),
        image_suffix=str(cfg["data"].get("image_suffix", ".png")),
        annotation_suffix=str(cfg["data"].get("annotation_suffix", ".json")),
        target_label=str(cfg["data"].get("target_label", "camou")),
        ignore_labels=tuple(cfg["data"].get("ignore_labels", ["skin"])),
        strict_non_empty_mask=True,
    )
    dataset = LabelMeCamouflageDataset(dataset_cfg)

    n_total = len(dataset)
    n_subset = min(24, n_total)
    if n_subset < 4:
        raise RuntimeError(f"Dataset subset too small for sanity check: {n_subset}")

    subset = Subset(dataset, list(range(n_subset)))
    n_train = max(2, int(0.75 * n_subset))
    n_train = min(n_train, n_subset - 1)
    train_subset = Subset(subset, list(range(n_train)))
    val_subset = Subset(subset, list(range(n_train, n_subset)))

    batch_size = min(int(cfg.get("hyperparameters", {}).get("batch_size", 4)), 4)

    image_size = int(cfg.get("hcas_specific", {}).get("image_size", 256))
    train_augmenter = build_augmentations(cfg.get("augmentations", {}))
    aug_strength = None
    if train_augmenter is not None and hasattr(train_augmenter, "set_epoch"):
        train_augmenter.set_epoch(1)
    if train_augmenter is not None and hasattr(train_augmenter, "get_strength_factor"):
        aug_strength = float(train_augmenter.get_strength_factor())

    train_collate_fn = build_resize_collate_fn(image_size=image_size, augmenter=train_augmenter)
    eval_collate_fn = build_resize_collate_fn(image_size=image_size)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=train_collate_fn,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=eval_collate_fn,
    )

    device = _resolve_device(cfg)

    generator = GeneratorUNet().to(device)
    discriminator = PatchDiscriminator().to(device)
    saliency_cfg = cfg.get("saliency_model", {})
    saliency_model = DeepGazeWrapper(
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
    criterion = HCASLoss(lambda_sal=float(cfg.get("hcas_specific", {}).get("lambda_sal", 15.0)))

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=float(cfg["hyperparameters"].get("learning_rate_G", 2e-4)),
        betas=(
            float(cfg["hyperparameters"].get("beta1", 0.5)),
            float(cfg["hyperparameters"].get("beta2", 0.999)),
        ),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=float(cfg["hyperparameters"].get("learning_rate_D", 2e-4)),
        betas=(
            float(cfg["hyperparameters"].get("beta1", 0.5)),
            float(cfg["hyperparameters"].get("beta2", 0.999)),
        ),
    )

    train_metrics = train_one_epoch(
        generator=generator,
        discriminator=discriminator,
        saliency_model=saliency_model,
        criterion=criterion,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        dataloader=train_loader,
        device=device,
        log_interval=0,
    )
    val_metrics = validate_one_epoch(
        generator=generator,
        discriminator=discriminator,
        saliency_model=saliency_model,
        criterion=criterion,
        dataloader=val_loader,
        device=device,
    )

    train_dict = train_metrics.to_dict()
    val_dict = val_metrics.to_dict()

    saliency_info = saliency_model.get_backend_info()

    print(f"device={device}")
    aug_label = "enabled" if train_augmenter is not None else "disabled"
    if aug_strength is not None:
        aug_label += f"(strength={aug_strength:.3f})"
    print("augmentations=" + aug_label)
    print(
        "saliency_backend="
        + json.dumps(
            {
                "requested": saliency_info.requested_backend,
                "active": saliency_info.active_backend,
                "model": saliency_info.model_name,
                "using_fallback": saliency_info.using_fallback,
                "fallback_reason": saliency_info.fallback_reason,
            },
            indent=2,
        )
    )
    print(f"subset_total={n_subset} train={len(train_subset)} val={len(val_subset)}")
    print("train_metrics=" + json.dumps(train_dict, indent=2))
    print("val_metrics=" + json.dumps(val_dict, indent=2))

    for name, value in {**train_dict, **val_dict}.items():
        if isinstance(value, float):
            if not np.isfinite(value):
                raise AssertionError(f"Metric {name} is not finite: {value}")

    if train_dict["num_batches"] <= 0 or train_dict["num_samples"] <= 0:
        raise AssertionError("Train loop did not consume any samples.")
    if val_dict["num_batches"] <= 0 or val_dict["num_samples"] <= 0:
        raise AssertionError("Validation loop did not consume any samples.")

    print("TRAINER SANITY CHECK PASSED")


if __name__ == "__main__":
    main()
