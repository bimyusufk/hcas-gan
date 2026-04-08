"""Training curriculum helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class LambdaScheduler:
    enabled: bool
    mode: str
    start_value: float
    end_value: float
    warmup_epochs: int

    def value(self, epoch: int) -> float:
        if not self.enabled:
            return float(self.end_value)

        e = max(1, int(epoch))
        w = max(1, int(self.warmup_epochs))
        t = min(float(e) / float(w), 1.0)

        mode = self.mode.lower()
        if mode == "cosine":
            alpha = 0.5 - 0.5 * math.cos(math.pi * t)
        elif mode == "step":
            alpha = 0.0 if t < 1.0 else 1.0
        else:
            alpha = t  # linear

        return float(self.start_value + (self.end_value - self.start_value) * alpha)


def build_lambda_scheduler(
    *,
    enabled: bool,
    mode: str,
    start_value: float,
    end_value: float,
    warmup_epochs: int,
) -> LambdaScheduler:
    return LambdaScheduler(
        enabled=bool(enabled),
        mode=str(mode),
        start_value=float(start_value),
        end_value=float(end_value),
        warmup_epochs=int(warmup_epochs),
    )
