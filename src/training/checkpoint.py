"""Checkpoint management for HCAS-GAN training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


@dataclass
class CheckpointData:
    """Checkpoint metadata and state snapshot."""

    epoch: int
    step: int
    generator_state: dict[str, Any]
    discriminator_state: dict[str, Any]
    optimizer_g_state: dict[str, Any]
    optimizer_d_state: dict[str, Any]
    scheduler_g_state: dict[str, Any] | None = None
    scheduler_d_state: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class CheckpointManager:
    """Save and load training checkpoints."""

    def __init__(self, checkpoint_dir: str | Path, keep_last_n: int = 3):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints.
            keep_last_n: Maximum number of checkpoints to keep (oldest deleted).
        """
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.keep_last_n = max(1, int(keep_last_n))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        epoch: int,
        step: int,
        generator: nn.Module,
        discriminator: nn.Module,
        optimizer_g: Optimizer,
        optimizer_d: Optimizer,
        scheduler_g: Any = None,
        scheduler_d: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save checkpoint to disk.

        Args:
            epoch: Current epoch number.
            step: Current step number.
            generator: Generator model.
            discriminator: Discriminator model.
            optimizer_g: Generator optimizer.
            optimizer_d: Discriminator optimizer.
            scheduler_g: Generator scheduler (optional).
            scheduler_d: Discriminator scheduler (optional).
            metadata: Additional metadata to store.

        Returns:
            Path to saved checkpoint file.
        """
        def _state_dict(module: nn.Module) -> dict[str, Any]:
            if isinstance(module, torch.nn.DataParallel):
                return module.module.state_dict()
            return module.state_dict()

        checkpoint_data = CheckpointData(
            epoch=int(epoch),
            step=int(step),
            generator_state=_state_dict(generator),
            discriminator_state=_state_dict(discriminator),
            optimizer_g_state=optimizer_g.state_dict(),
            optimizer_d_state=optimizer_d.state_dict(),
            scheduler_g_state=scheduler_g.state_dict() if scheduler_g is not None else None,
            scheduler_d_state=scheduler_d.state_dict() if scheduler_d is not None else None,
            metadata=metadata or {},
        )

        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}_step_{step:06d}.pt"

        torch.save(asdict(checkpoint_data), checkpoint_path)
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def load(
        self,
        checkpoint_path: str | Path,
        *,
        generator: nn.Module,
        discriminator: nn.Module,
        optimizer_g: Optimizer,
        optimizer_d: Optimizer,
        scheduler_g: Any = None,
        scheduler_d: Any = None,
        device: torch.device | None = None,
    ) -> CheckpointData:
        """Load checkpoint from disk.

        Args:
            checkpoint_path: Path to checkpoint file.
            generator: Generator model to load state into.
            discriminator: Discriminator model to load state into.
            optimizer_g: Generator optimizer to load state into.
            optimizer_d: Discriminator optimizer to load state into.
            scheduler_g: Generator scheduler to load state into (optional).
            scheduler_d: Discriminator scheduler to load state into (optional).
            device: Device to load checkpoint on (defaults to current device).

        Returns:
            CheckpointData object containing loaded metadata.
        """
        checkpoint_path = Path(checkpoint_path).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if device is None:
            device = torch.device("cpu")

        checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if isinstance(checkpoint_data, dict):
            checkpoint_data_obj = CheckpointData(
                epoch=int(checkpoint_data["epoch"]),
                step=int(checkpoint_data["step"]),
                generator_state=checkpoint_data.get("generator_state", {}),
                discriminator_state=checkpoint_data.get("discriminator_state", {}),
                optimizer_g_state=checkpoint_data.get("optimizer_g_state", {}),
                optimizer_d_state=checkpoint_data.get("optimizer_d_state", {}),
                scheduler_g_state=checkpoint_data.get("scheduler_g_state"),
                scheduler_d_state=checkpoint_data.get("scheduler_d_state"),
                metadata=checkpoint_data.get("metadata", {}),
            )
        else:
            checkpoint_data_obj = checkpoint_data

        def _load(module: nn.Module, state: dict[str, Any]) -> None:
            target = module.module if isinstance(module, torch.nn.DataParallel) else module
            target.load_state_dict(state)

        _load(generator, checkpoint_data_obj.generator_state)
        _load(discriminator, checkpoint_data_obj.discriminator_state)
        optimizer_g.load_state_dict(checkpoint_data_obj.optimizer_g_state)
        optimizer_d.load_state_dict(checkpoint_data_obj.optimizer_d_state)

        if scheduler_g is not None and checkpoint_data_obj.scheduler_g_state is not None:
            scheduler_g.load_state_dict(checkpoint_data_obj.scheduler_g_state)

        if scheduler_d is not None and checkpoint_data_obj.scheduler_d_state is not None:
            scheduler_d.load_state_dict(checkpoint_data_obj.scheduler_d_state)

        return checkpoint_data_obj

    def find_latest_checkpoint(self) -> Path | None:
        """Find the most recent checkpoint in the directory."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
        return checkpoints[-1] if checkpoints else None

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond keep_last_n limit."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if len(checkpoints) > self.keep_last_n:
            for old_checkpoint in checkpoints[: -self.keep_last_n]:
                old_checkpoint.unlink()
