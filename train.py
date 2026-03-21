"""HCAS-GAN training entrypoint."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from src.data import (
    DatasetConfig,
    LabelMeCamouflageDataset,
    build_augmentations,
    build_resize_collate_fn,
    split_dataset,
)
from src.models.deepgaze_wrapper import DeepGazeWrapper
from src.models.discriminator import PatchDiscriminator
from src.models.generator import GeneratorUNet
from src.training.checkpoint import CheckpointManager
from src.training.loss import HCASLoss
from src.training.scheduler import build_scheduler, get_learning_rate, step_scheduler
from src.training.tensorboard import build_summary_writer
from src.training.trainer import train_one_epoch, validate_one_epoch


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_runtime_hardware(config: dict) -> tuple[torch.device, int]:
    hardware_cfg = config.get("hardware", {})
    preferred = str(hardware_cfg.get("device", "auto")).lower()

    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0

    if preferred == "cpu":
        return torch.device("cpu"), 0

    if preferred in ("auto", "cuda") and cuda_available:
        return torch.device("cuda"), gpu_count

    return torch.device("cpu"), 0


def should_use_multi_gpu(config: dict, device: torch.device, gpu_count: int) -> bool:
    if device.type != "cuda":
        return False

    multi_gpu_cfg = config.get("hardware", {}).get("multi_gpu", "auto")
    if isinstance(multi_gpu_cfg, bool):
        return multi_gpu_cfg and gpu_count >= 2

    policy = str(multi_gpu_cfg).lower()
    if policy == "off":
        return False
    if policy == "on":
        return gpu_count >= 2
    return gpu_count >= 2


def log_visible_cuda_devices(gpu_count: int) -> None:
    if gpu_count <= 0:
        print("[HCAS-GAN] No CUDA devices visible.")
        return

    device_names = []
    for idx in range(gpu_count):
        try:
            device_names.append(torch.cuda.get_device_name(idx))
        except Exception as exc:
            device_names.append(f"<unavailable:{type(exc).__name__}>")
    print(f"[HCAS-GAN] Visible CUDA devices ({gpu_count}): {device_names}")


def maybe_wrap_data_parallel(model: nn.Module, use_multi_gpu: bool, gpu_count: int) -> nn.Module:
    if use_multi_gpu and gpu_count >= 2:
        device_ids = list(range(gpu_count))
        return nn.DataParallel(model, device_ids=device_ids, output_device=0, dim=0)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HCAS-GAN")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override (defaults to hyperparameters.epochs in config)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=0,
        help="Print train-step logs every N batches (0 disables)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save/load checkpoints (overrides config.checkpoint.dir)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Save checkpoint every N epochs (overrides config.checkpoint.interval)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (or 'latest' to auto-find). Overrides config.checkpoint.resume_from",
    )
    parser.add_argument(
        "--no-scheduler",
        action="store_true",
        help="Disable learning-rate scheduler",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default=None,
        help="TensorBoard base log directory (overrides config.tensorboard.log_dir)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    config = load_config((root / args.config).resolve())

    print(f"[HCAS-GAN] Experiment: {config['experiment']['name']}")

    seed = int(config.get("experiment", {}).get("seed", 42))
    set_seed(seed)

    device, gpu_count = resolve_runtime_hardware(config)
    multi_gpu = should_use_multi_gpu(config, device, gpu_count)
    print(
        f"[HCAS-GAN] Device: {device} | cuda_available={torch.cuda.is_available()} | gpu_count={gpu_count} | multi_gpu={multi_gpu}"
    )
    log_visible_cuda_devices(gpu_count)

    total_epochs = int(args.epochs or config.get("hyperparameters", {}).get("epochs", 1))

    checkpoint_cfg = config.get("checkpoint", {})
    checkpoint_dir = str(args.checkpoint_dir or checkpoint_cfg.get("dir", "./checkpoints"))
    checkpoint_interval = int(
        args.checkpoint_interval
        if args.checkpoint_interval is not None
        else checkpoint_cfg.get("interval", 5)
    )
    checkpoint_keep_last_n = int(checkpoint_cfg.get("keep_last_n", 3))
    resume_from = args.resume_from if args.resume_from is not None else checkpoint_cfg.get("resume_from")

    scheduler_cfg = config.get("scheduler", {})
    scheduler_enabled = (not args.no_scheduler) and bool(scheduler_cfg.get("enabled", True))
    scheduler_type = str(scheduler_cfg.get("type", "cosine"))
    scheduler_kwargs_raw = scheduler_cfg.get("kwargs", {})
    scheduler_kwargs = dict(scheduler_kwargs_raw or {})
    if scheduler_type.lower() == "cosine" and "t_max" not in scheduler_kwargs:
        scheduler_kwargs["t_max"] = total_epochs

    tensorboard_cfg = config.get("tensorboard", {})
    tensorboard_enabled = (not args.no_tensorboard) and bool(tensorboard_cfg.get("enabled", True))
    tensorboard_dir = Path(str(args.tensorboard_dir or tensorboard_cfg.get("log_dir", "./runs")))
    if not tensorboard_dir.is_absolute():
        tensorboard_dir = (root / tensorboard_dir).resolve()
    tensorboard_flush_secs = int(tensorboard_cfg.get("flush_secs", 30))
    run_name_cfg = tensorboard_cfg.get("run_name")
    if run_name_cfg is None or str(run_name_cfg).strip() == "":
        tensorboard_run_name = f"{config['experiment']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        tensorboard_run_name = str(run_name_cfg)

    dataset_cfg = DatasetConfig(
        annotations_dir=str((root / config["data"]["annotations_dir"]).resolve()),
        image_suffix=str(config["data"].get("image_suffix", ".png")),
        annotation_suffix=str(config["data"].get("annotation_suffix", ".json")),
        target_label=str(config["data"].get("target_label", "camou")),
        ignore_labels=tuple(config["data"].get("ignore_labels", ["skin"])),
        strict_non_empty_mask=True,
    )
    dataset = LabelMeCamouflageDataset(dataset_cfg)

    train_set, val_set, test_set = split_dataset(
        dataset,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=seed,
    )

    image_size = int(config.get("hcas_specific", {}).get("image_size", 256))
    augmentations_cfg = config.get("augmentations", {})
    train_augmenter = build_augmentations(augmentations_cfg)

    train_collate_fn = build_resize_collate_fn(
        image_size=image_size,
        augmenter=train_augmenter,
    )
    eval_collate_fn = build_resize_collate_fn(image_size=image_size)

    batch_size = int(config.get("hyperparameters", {}).get("batch_size", 16))
    num_workers = int(config.get("hardware", {}).get("num_workers", 0))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=train_collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=eval_collate_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=eval_collate_fn,
    )

    print(
        "[HCAS-GAN] Train augmentations: "
        + ("enabled" if train_augmenter is not None else "disabled")
    )
    if train_augmenter is not None and hasattr(train_augmenter, "get_strength_factor"):
        aug_strength = float(train_augmenter.get_strength_factor())
        print(f"[HCAS-GAN] Initial augmentation strength={aug_strength:.3f}")

    generator = maybe_wrap_data_parallel(GeneratorUNet().to(device), multi_gpu, gpu_count)
    discriminator = maybe_wrap_data_parallel(PatchDiscriminator().to(device), multi_gpu, gpu_count)

    saliency_cfg = config.get("saliency_model", {})
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
    saliency_info = saliency_model.get_backend_info()
    print(
        f"[HCAS-GAN] Saliency backend: active={saliency_info.active_backend} "
        f"requested={saliency_info.requested_backend} model={saliency_info.model_name}"
    )
    if saliency_info.using_fallback and saliency_info.fallback_reason:
        print(f"[HCAS-GAN] Saliency fallback reason: {saliency_info.fallback_reason}")

    criterion = HCASLoss(lambda_sal=float(config.get("hcas_specific", {}).get("lambda_sal", 15.0)))
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=float(config["hyperparameters"].get("learning_rate_G", 2e-4)),
        betas=(
            float(config["hyperparameters"].get("beta1", 0.5)),
            float(config["hyperparameters"].get("beta2", 0.999)),
        ),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=float(config["hyperparameters"].get("learning_rate_D", 2e-4)),
        betas=(
            float(config["hyperparameters"].get("beta1", 0.5)),
            float(config["hyperparameters"].get("beta2", 0.999)),
        ),
    )

    scheduler_g = None
    scheduler_d = None
    if scheduler_enabled:
        scheduler_g = build_scheduler(
            optimizer_g,
            scheduler_type=scheduler_type,
            **scheduler_kwargs,
        )
        scheduler_d = build_scheduler(
            optimizer_d,
            scheduler_type=scheduler_type,
            **scheduler_kwargs,
        )
        print(f"[HCAS-GAN] Scheduler: enabled type={scheduler_type} kwargs={scheduler_kwargs}")
    else:
        print("[HCAS-GAN] Scheduler: disabled")

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        keep_last_n=checkpoint_keep_last_n,
    )
    print(
        f"[HCAS-GAN] Checkpoint: dir={checkpoint_manager.checkpoint_dir} "
        f"interval={checkpoint_interval} keep_last_n={checkpoint_keep_last_n}"
    )

    writer = build_summary_writer(
        enabled=tensorboard_enabled,
        log_dir=tensorboard_dir,
        run_name=tensorboard_run_name,
        flush_secs=tensorboard_flush_secs,
    )
    if writer is not None:
        writer.add_text("experiment/name", str(config.get("experiment", {}).get("name", "HCAS-GAN")))
        writer.add_text("runtime/device", str(device))
        writer.add_text("scheduler/type", scheduler_type if scheduler_enabled else "disabled")
        writer.add_text("saliency/backend", saliency_info.active_backend)
        print(f"[HCAS-GAN] TensorBoard: enabled log_dir={writer.log_dir}")
    else:
        print("[HCAS-GAN] TensorBoard: disabled")

    start_epoch = 1
    if resume_from:
        resume_path = str(resume_from)
        if resume_path == "latest":
            latest = checkpoint_manager.find_latest_checkpoint()
            if latest is None:
                print("[HCAS-GAN] No checkpoint found. Starting from scratch.")
            else:
                resume_path = str(latest)
        
        if resume_path != "latest":
            print(f"[HCAS-GAN] Resuming from: {resume_path}")
            try:
                ckpt_data = checkpoint_manager.load(
                    resume_path,
                    generator=generator,
                    discriminator=discriminator,
                    optimizer_g=optimizer_g,
                    optimizer_d=optimizer_d,
                    scheduler_g=scheduler_g,
                    scheduler_d=scheduler_d,
                    device=device,
                )
                start_epoch = ckpt_data.epoch + 1
                print(f"[HCAS-GAN] Resumed from epoch {ckpt_data.epoch}, resuming at epoch {start_epoch}")
                if writer is not None:
                    writer.add_text("checkpoint/resume", f"resumed_from={resume_path} epoch={ckpt_data.epoch}")
            except Exception as e:
                print(f"[HCAS-GAN] Failed to resume from {resume_path}: {e}", file=sys.stderr)
                print("[HCAS-GAN] Starting from scratch.")
                start_epoch = 1

    print(
        f"[HCAS-GAN] Dataset split: train={len(train_set)} val={len(val_set)} test={len(test_set)} "
        f"| image_size={image_size} | batch_size={batch_size} | epochs={total_epochs}"
    )

    for epoch in range(start_epoch, total_epochs + 1):
        if train_augmenter is not None and hasattr(train_augmenter, "set_epoch"):
            train_augmenter.set_epoch(epoch)

        aug_strength = None
        if train_augmenter is not None and hasattr(train_augmenter, "get_strength_factor"):
            aug_strength = float(train_augmenter.get_strength_factor())

        lr_g = get_learning_rate(optimizer_g)
        lr_d = get_learning_rate(optimizer_d)
        
        train_metrics = train_one_epoch(
            generator=generator,
            discriminator=discriminator,
            saliency_model=saliency_model,
            criterion=criterion,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            dataloader=train_loader,
            device=device,
            log_interval=int(args.log_interval),
        )
        val_metrics = validate_one_epoch(
            generator=generator,
            discriminator=discriminator,
            saliency_model=saliency_model,
            criterion=criterion,
            dataloader=val_loader,
            device=device,
        )

        step_scheduler(scheduler_g)
        step_scheduler(scheduler_d)

        aug_part = f" aug={aug_strength:.3f}" if aug_strength is not None else ""
        print(
            f"[epoch {epoch}/{total_epochs}] "
            f"train_g={train_metrics.g_total:.4f} train_d={train_metrics.d_total:.4f} "
            f"val_g={val_metrics.g_total:.4f} val_d={val_metrics.d_total:.4f} "
            f"lr_g={lr_g:.6f} lr_d={lr_d:.6f}{aug_part}"
        )

        if writer is not None:
            writer.add_scalar("loss/train/g_total", train_metrics.g_total, epoch)
            writer.add_scalar("loss/train/d_total", train_metrics.d_total, epoch)
            writer.add_scalar("loss/train/g_adv", train_metrics.g_adv, epoch)
            writer.add_scalar("loss/train/g_sal", train_metrics.g_sal, epoch)
            writer.add_scalar("loss/val/g_total", val_metrics.g_total, epoch)
            writer.add_scalar("loss/val/d_total", val_metrics.d_total, epoch)
            writer.add_scalar("loss/val/g_adv", val_metrics.g_adv, epoch)
            writer.add_scalar("loss/val/g_sal", val_metrics.g_sal, epoch)
            writer.add_scalar("lr/g", lr_g, epoch)
            writer.add_scalar("lr/d", lr_d, epoch)
            if aug_strength is not None:
                writer.add_scalar("augmentation/strength", aug_strength, epoch)
            writer.add_scalar("timing/train_epoch_sec", train_metrics.duration_sec, epoch)
            writer.add_scalar("timing/val_epoch_sec", val_metrics.duration_sec, epoch)

        if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
            ckpt_path = checkpoint_manager.save(
                epoch=epoch,
                step=epoch * len(train_loader),
                generator=generator,
                discriminator=discriminator,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                scheduler_g=scheduler_g,
                scheduler_d=scheduler_d,
                metadata={
                    "train_g_loss": train_metrics.g_total,
                    "val_g_loss": val_metrics.g_total,
                    "train_d_loss": train_metrics.d_total,
                    "val_d_loss": val_metrics.d_total,
                    "lr_g": lr_g,
                    "lr_d": lr_d,
                    "augmentation_strength": aug_strength,
                },
            )
            print(f"[HCAS-GAN] Checkpoint saved: {ckpt_path.name}")
            if writer is not None:
                writer.add_text("checkpoint/saved", f"epoch={epoch} path={ckpt_path.name}", epoch)

    test_metrics = validate_one_epoch(
        generator=generator,
        discriminator=discriminator,
        saliency_model=saliency_model,
        criterion=criterion,
        dataloader=test_loader,
        device=device,
    )
    print(
        "[final test] "
        f"g_total={test_metrics.g_total:.4f} d_total={test_metrics.d_total:.4f} "
        f"samples={test_metrics.num_samples}"
    )
    if writer is not None:
        writer.add_scalar("loss/test/g_total", test_metrics.g_total, total_epochs)
        writer.add_scalar("loss/test/d_total", test_metrics.d_total, total_epochs)
        writer.flush()
        writer.close()
    print("[HCAS-GAN] Training complete!")


if __name__ == "__main__":
    main()


