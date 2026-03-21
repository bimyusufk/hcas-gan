"""SSIM metric helpers for HCAS-GAN.

Provides lightweight wrappers around ``skimage.metrics.structural_similarity``
with batched PyTorch tensor input support.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from skimage.metrics import structural_similarity


def _to_numpy_chw(image: torch.Tensor) -> np.ndarray:
	if not isinstance(image, torch.Tensor):
		raise TypeError(f"image must be torch.Tensor, got: {type(image).__name__}")
	if image.ndim != 3:
		raise ValueError(f"image must be [C,H,W], got shape: {tuple(image.shape)}")
	if image.shape[0] not in (1, 3):
		raise ValueError(f"image channel must be 1 or 3, got: {int(image.shape[0])}")

	arr = image.detach().float().cpu().clamp(0.0, 1.0).numpy()
	return arr


def _ssim_single(pred: torch.Tensor, target: torch.Tensor) -> float:
	pred_np = _to_numpy_chw(pred)
	target_np = _to_numpy_chw(target)
	if pred_np.shape != target_np.shape:
		raise ValueError(
			"pred/target shapes must match. "
			f"Got {pred_np.shape} vs {target_np.shape}."
		)

	# skimage expects HWC for multi-channel input.
	if pred_np.shape[0] == 1:
		pred_2d = pred_np[0]
		target_2d = target_np[0]
		score = structural_similarity(
			pred_2d,
			target_2d,
			data_range=1.0,
		)
	else:
		pred_hwc = np.transpose(pred_np, (1, 2, 0))
		target_hwc = np.transpose(target_np, (1, 2, 0))
		score = structural_similarity(
			pred_hwc,
			target_hwc,
			data_range=1.0,
			channel_axis=-1,
		)
	return float(score)


def compute_ssim_batch(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, list[float]]:
	"""Compute SSIM for a batch of images.

	Args:
		pred: Tensor of shape [N,C,H,W], expected range [0,1].
		target: Tensor of shape [N,C,H,W], expected range [0,1].

	Returns:
		mean_ssim: Average SSIM over the batch.
		per_sample: Per-sample SSIM values.
	"""
	if pred.ndim != 4 or target.ndim != 4:
		raise ValueError(
			f"pred/target must be [N,C,H,W], got {tuple(pred.shape)} and {tuple(target.shape)}"
		)
	if pred.shape != target.shape:
		raise ValueError(
			f"pred/target shapes must match, got {tuple(pred.shape)} vs {tuple(target.shape)}"
		)
	if pred.shape[0] == 0:
		raise ValueError("Empty batch is not allowed.")

	scores = [_ssim_single(pred[i], target[i]) for i in range(pred.shape[0])]
	mean_score = float(np.mean(scores))
	return mean_score, [float(s) for s in scores]


def summarize_ssim(scores: Iterable[float]) -> dict[str, float]:
	vals = [float(x) for x in scores]
	if len(vals) == 0:
		return {"mean": 0.0, "min": 0.0, "max": 0.0}
	return {
		"mean": float(np.mean(vals)),
		"min": float(np.min(vals)),
		"max": float(np.max(vals)),
	}
