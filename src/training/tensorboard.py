"""TensorBoard logging utilities for HCAS-GAN training."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_summary_writer(
    *,
    enabled: bool,
    log_dir: str | Path,
    run_name: str,
    flush_secs: int = 30,
) -> Any | None:
    """Build a TensorBoard SummaryWriter if available.

    Returns None when disabled or tensorboard dependency is unavailable.
    """
    if not enabled:
        return None

    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        print(f"[HCAS-GAN] TensorBoard disabled (unavailable): {type(exc).__name__}: {exc}")
        return None

    base_dir = Path(log_dir).resolve()
    writer_dir = base_dir / run_name
    writer_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(writer_dir), flush_secs=max(1, int(flush_secs)))
    return writer
