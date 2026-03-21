"""LPIPS metric helpers for HCAS-GAN.

Uses the optional ``lpips`` package when available.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LPIPSResult:
	mean: float
	per_sample: list[float]
	available: bool
	backend: str


def _to_lpips_range(x: torch.Tensor) -> torch.Tensor:
	"""Convert tensor from [0,1] to [-1,1] expected by LPIPS."""
	if x.ndim != 4:
		raise ValueError(f"Expected tensor [N,C,H,W], got: {tuple(x.shape)}")
	if x.shape[1] not in (1, 3):
		raise ValueError(f"Channel must be 1 or 3, got: {int(x.shape[1])}")

	y = x.float().clamp(0.0, 1.0)
	if y.shape[1] == 1:
		y = y.repeat(1, 3, 1, 1)
	return (y * 2.0) - 1.0


def compute_lpips_batch(
	pred: torch.Tensor,
	target: torch.Tensor,
	*,
	net: str = "alex",
	device: torch.device | None = None,
) -> LPIPSResult:
	"""Compute LPIPS score for a batch.

	Returns availability status when ``lpips`` dependency is missing instead of crashing.
	"""
	if pred.shape != target.shape:
		raise ValueError(
			f"pred/target shapes must match, got {tuple(pred.shape)} vs {tuple(target.shape)}"
		)
	if pred.ndim != 4:
		raise ValueError(f"Expected [N,C,H,W], got: {tuple(pred.shape)}")
	if pred.shape[0] == 0:
		raise ValueError("Empty batch is not allowed.")

	try:
		import lpips  # type: ignore
	except Exception:
		return LPIPSResult(
			mean=0.0,
			per_sample=[0.0 for _ in range(pred.shape[0])],
			available=False,
			backend="unavailable",
		)

	if device is None:
		device = pred.device

	model = lpips.LPIPS(net=net).to(device)
	model.eval()
	for p in model.parameters():
		p.requires_grad = False

	x = _to_lpips_range(pred).to(device)
	y = _to_lpips_range(target).to(device)

	with torch.no_grad():
		vals = model(x, y)

	vals = vals.reshape(-1).detach().cpu().tolist()
	vals = [float(v) for v in vals]
	mean_val = float(sum(vals) / max(1, len(vals)))
	return LPIPSResult(
		mean=mean_val,
		per_sample=vals,
		available=True,
		backend=f"lpips:{net}",
	)
