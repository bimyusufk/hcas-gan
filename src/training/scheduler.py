"""Learning-rate schedulers for HCAS-GAN training."""

from __future__ import annotations

from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, StepLR


def build_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "step",
    **kwargs: Any,
) -> Any:
    """Build a learning-rate scheduler.

    Args:
        optimizer: PyTorch optimizer to schedule.
        scheduler_type: One of ["step", "cosine", "lambda"]. Defaults to "step".
        **kwargs: Scheduler-specific keyword arguments.

    Returns:
        PyTorch scheduler instance or None.
    """
    scheduler_type = str(scheduler_type).lower()

    if scheduler_type == "step":
        step_size = int(kwargs.get("step_size", 10))
        gamma = float(kwargs.get("gamma", 0.5))
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    if scheduler_type == "cosine":
        t_max = int(kwargs.get("t_max", 100))
        eta_min = float(kwargs.get("eta_min", 0.0))
        return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    if scheduler_type == "lambda":
        lr_lambda = kwargs.get("lr_lambda", lambda epoch: 1.0)
        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    return None


def step_scheduler(scheduler: Any) -> None:
    """Step a scheduler to the next learning-rate regime."""
    if scheduler is not None:
        scheduler.step()


def get_learning_rate(optimizer: Optimizer) -> float:
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return float(param_group["lr"])
    return 0.0
