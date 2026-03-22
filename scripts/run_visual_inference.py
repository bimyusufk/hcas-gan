"""Run visual inference for HCAS-GAN checkpoints.

Outputs:
1) Comparison panel (environment vs generated pattern vs composite).
2) Flat 1x1 generated pattern image(s).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    DatasetConfig,
    LabelMeCamouflageDataset,
    build_resize_collate_fn,
    reproducible_split_indices,
)
from src.models import GeneratorUNet


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HCAS-GAN visual inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--annotations-dir", type=str, default="./camo", help="Dataset annotation directory")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--seed", type=int, default=42, help="Split seed")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio")
    parser.add_argument("--image-size", type=int, default=256, help="Inference image size")
    parser.add_argument("--batch-size", type=int, default=6, help="Batch size for sampling")
    parser.add_argument("--num-samples", type=int, default=6, help="How many samples to export")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Inference device")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="inference_outputs",
        help="Base output directory",
    )
    parser.add_argument(
        "--output-comparison",
        type=str,
        default="comparison_3xN.png",
        help="Filename for comparison image",
    )
    parser.add_argument(
        "--output-pattern-flat",
        type=str,
        default="pattern_flat_1x1_sample01.png",
        help="Filename for first sample flat 1x1 pattern",
    )
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_generator_from_checkpoint(checkpoint_path: Path, device: torch.device) -> GeneratorUNet:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    gen_state = ckpt.get("generator_state")
    if not isinstance(gen_state, dict):
        raise RuntimeError("Checkpoint does not contain 'generator_state'.")

    model = GeneratorUNet().to(device)
    model.load_state_dict(gen_state, strict=True)
    model.eval()
    return model


def _make_subset(dataset: LabelMeCamouflageDataset, split: str, *, seed: int, val_ratio: float, test_ratio: float) -> Subset:
    train_idx, val_idx, test_idx = reproducible_split_indices(
        len(dataset),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    indices_map = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }
    return Subset(dataset, indices_map[split])


def _to_bgr_u8(x: torch.Tensor) -> np.ndarray:
    # x: [3,H,W] range [0,1]
    arr = x.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _to_mask_vis(mask: torch.Tensor) -> np.ndarray:
    # mask: [1,H,W] {0,1}
    m = mask.detach().cpu().clamp(0.0, 1.0)[0].numpy()
    m = (m * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)


def _annotate(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(
        out,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return out


def _compose(background: torch.Tensor, pattern: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (background * (1.0 - mask)) + (pattern * mask)


def main() -> None:
    args = _parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = _resolve_device(args.device)

    dataset_cfg = DatasetConfig(
        annotations_dir=str(Path(args.annotations_dir).resolve()),
        image_suffix=".png",
        annotation_suffix=".json",
        target_label="camou",
        ignore_labels=("skin", "weapon"),
        strict_non_empty_mask=True,
    )
    dataset = LabelMeCamouflageDataset(dataset_cfg)
    split_subset = _make_subset(
        dataset,
        args.split,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    if len(split_subset) == 0:
        raise RuntimeError(f"Split '{args.split}' is empty.")

    collate = build_resize_collate_fn(image_size=int(args.image_size))

    generator = _load_generator_from_checkpoint(checkpoint_path, device)

    wanted = max(1, int(args.num_samples))
    collected: list[dict[str, torch.Tensor | str]] = []
    skipped = 0
    for i in range(len(split_subset)):
        if len(collected) >= wanted:
            break
        try:
            collected.append(split_subset[i])
        except Exception:
            skipped += 1

    if len(collected) == 0:
        raise RuntimeError(
            "No readable samples found in split. "
            "Please verify dataset image paths and encoding compatibility."
        )

    if len(collected) < wanted:
        print(
            f"[visual inference] Warning: requested {wanted} samples but only {len(collected)} readable samples found."
        )

    batch = collate(collected)
    images = batch["image"].to(device=device, dtype=torch.float32)
    masks = batch["mask"].to(device=device, dtype=torch.float32)

    with torch.no_grad():
        patterns = generator(images)
        composites = _compose(images, patterns, masks)

    n = int(images.shape[0])

    rows: list[np.ndarray] = []
    for i in range(n):
        env_img = _annotate(_to_bgr_u8(images[i]), "Environment")
        mask_img = _annotate(_to_mask_vis(masks[i]), "Mask")
        pat_img = _annotate(_to_bgr_u8(patterns[i]), "Generated Pattern")
        comp_img = _annotate(_to_bgr_u8(composites[i]), "Composite")

        row = np.concatenate([env_img, mask_img, pat_img, comp_img], axis=1)
        rows.append(row)

    comparison = np.concatenate(rows, axis=0)

    run_dir = Path(args.output_dir).resolve() / checkpoint_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = run_dir / args.output_comparison
    cv2.imwrite(str(comparison_path), comparison)

    # Requested flat 1x1 pattern image: take first generated pattern (already square HxW)
    pattern_flat_path = run_dir / args.output_pattern_flat
    first_pattern = _to_bgr_u8(patterns[0])
    cv2.imwrite(str(pattern_flat_path), first_pattern)

    # Also export all generated flat patterns for convenience
    for i in range(n):
        p = _to_bgr_u8(patterns[i])
        cv2.imwrite(str(run_dir / f"pattern_flat_1x1_sample{i+1:02d}.png"), p)

    print(f"[visual inference] checkpoint={checkpoint_path.name}")
    print(f"[visual inference] split={args.split} samples={n} skipped={skipped} device={device}")
    print(f"[visual inference] comparison={comparison_path}")
    print(f"[visual inference] pattern_flat_1x1={pattern_flat_path}")


if __name__ == "__main__":
    main()
