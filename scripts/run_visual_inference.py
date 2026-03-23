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
        "--compose-mode",
        type=str,
        default="tile",
        choices=["direct", "tile"],
        help="direct: old behavior (full-image pattern masked). tile: use 1x1 tile source then repeat to canvas before masking.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=64,
        help="Tile source size used when compose-mode=tile.",
    )
    parser.add_argument(
        "--no-random-tiling",
        action="store_true",
        help="Disable random flip/rotate/shift/color jitter per tile block.",
    )
    parser.add_argument(
        "--no-color-match",
        action="store_true",
        help="Disable local color transfer from environment neighborhood to pattern.",
    )
    parser.add_argument(
        "--ring-width",
        type=int,
        default=11,
        help="Neighborhood ring width used for local color matching.",
    )
    parser.add_argument(
        "--feather-kernel",
        type=int,
        default=15,
        help="Gaussian kernel size for soft mask edge (odd number preferred).",
    )
    parser.add_argument(
        "--feather-sigma",
        type=float,
        default=3.0,
        help="Gaussian sigma for mask feathering.",
    )
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


def _ensure_odd(v: int) -> int:
    x = max(1, int(v))
    return x if (x % 2 == 1) else (x + 1)


def _extract_center_tile(pattern: torch.Tensor, tile_size: int) -> torch.Tensor:
    """Extract a center square tile from [N,3,H,W] pattern tensor."""
    if pattern.ndim != 4:
        raise ValueError(f"pattern must be [N,3,H,W], got {tuple(pattern.shape)}")
    h, w = int(pattern.shape[2]), int(pattern.shape[3])
    t = max(4, min(int(tile_size), h, w))
    y0 = (h - t) // 2
    x0 = (w - t) // 2
    return pattern[:, :, y0 : y0 + t, x0 : x0 + t]


def _tile_to_canvas(
    tile: torch.Tensor,
    out_h: int,
    out_w: int,
    *,
    randomize: bool,
    seed: int,
) -> torch.Tensor:
    """Repeat tile [N,3,t,t] to full canvas [N,3,out_h,out_w] with optional randomization."""
    if tile.ndim != 4:
        raise ValueError(f"tile must be [N,3,t,t], got {tuple(tile.shape)}")
    n, c, t_h, t_w = tile.shape

    canvas = torch.zeros((n, c, out_h, out_w), dtype=tile.dtype, device=tile.device)
    for bi in range(n):
        rng = np.random.default_rng(seed + bi)
        y = 0
        while y < out_h:
            x = 0
            while x < out_w:
                patch = tile[bi]
                if randomize:
                    if float(rng.random()) < 0.5:
                        patch = torch.flip(patch, dims=(1,))
                    if float(rng.random()) < 0.5:
                        patch = torch.flip(patch, dims=(2,))
                    k = int(rng.integers(0, 4))
                    if k > 0:
                        patch = torch.rot90(patch, k=k, dims=(1, 2))

                    shift_y = int(rng.integers(0, max(1, t_h)))
                    shift_x = int(rng.integers(0, max(1, t_w)))
                    patch = torch.roll(patch, shifts=(shift_y, shift_x), dims=(1, 2))

                    gain = float(rng.uniform(0.92, 1.08))
                    bias = float(rng.uniform(-0.03, 0.03))
                    patch = torch.clamp((patch * gain) + bias, 0.0, 1.0)

                h_slice = min(t_h, out_h - y)
                w_slice = min(t_w, out_w - x)
                canvas[bi, :, y : y + h_slice, x : x + w_slice] = patch[:, :h_slice, :w_slice]
                x += t_w
            y += t_h
    return canvas


def _feather_mask(mask: torch.Tensor, *, kernel_size: int, sigma: float) -> torch.Tensor:
    """Soft edge alpha mask using Gaussian blur per sample."""
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask must be [N,1,H,W], got {tuple(mask.shape)}")

    k = _ensure_odd(kernel_size)
    s = max(0.1, float(sigma))
    alpha_np = []
    for i in range(mask.shape[0]):
        m = mask[i, 0].detach().float().cpu().clamp(0.0, 1.0).numpy()
        if k > 1:
            m = cv2.GaussianBlur(m, (k, k), sigmaX=s, sigmaY=s)
        m = np.clip(m, 0.0, 1.0)
        alpha_np.append(m)
    alpha = torch.from_numpy(np.stack(alpha_np, axis=0)).unsqueeze(1).to(mask.device, dtype=mask.dtype)
    return alpha


def _local_color_transfer(
    pattern: torch.Tensor,
    background: torch.Tensor,
    mask: torch.Tensor,
    *,
    ring_width: int,
) -> torch.Tensor:
    """Match pattern color stats to local background around mask boundary."""
    if pattern.shape != background.shape:
        raise ValueError("pattern and background shapes must match")
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must be [N,1,H,W]")

    rw = max(1, int(ring_width))
    kernel = np.ones((rw, rw), dtype=np.uint8)
    out_samples: list[torch.Tensor] = []

    for bi in range(pattern.shape[0]):
        p = pattern[bi].detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
        b = background[bi].detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
        m = (mask[bi, 0].detach().float().cpu().numpy() > 0.5).astype(np.uint8)

        if int(m.sum()) == 0:
            out_samples.append(pattern[bi].detach().cpu())
            continue

        dil = cv2.dilate(m, kernel, iterations=1)
        ring = (dil > 0) & (m == 0)
        if int(ring.sum()) < 32:
            ring = np.ones_like(m, dtype=bool)

        m_bool = m > 0
        p_sel = p[m_bool]
        b_sel = b[ring]
        if p_sel.size == 0 or b_sel.size == 0:
            out_samples.append(pattern[bi].detach().cpu())
            continue

        p_mean = p_sel.mean(axis=0)
        p_std = p_sel.std(axis=0) + 1e-5
        b_mean = b_sel.mean(axis=0)
        b_std = b_sel.std(axis=0) + 1e-5

        p_adj = ((p - p_mean) / p_std) * b_std + b_mean
        p_adj = np.clip(p_adj, 0.0, 1.0)

        out = p.copy()
        out[m_bool] = p_adj[m_bool]

        out_t = torch.from_numpy(out).permute(2, 0, 1).float()
        out_samples.append(out_t)

    out_batch = torch.stack(out_samples, dim=0).to(pattern.device, dtype=pattern.dtype)
    return out_batch


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
        patterns_direct = generator(images)
        if args.compose_mode == "tile":
            tile_sources = _extract_center_tile(patterns_direct, int(args.tile_size))
            patterns_used = _tile_to_canvas(
                tile_sources,
                int(images.shape[2]),
                int(images.shape[3]),
                randomize=(not bool(args.no_random_tiling)),
                seed=int(args.seed),
            )
        else:
            tile_sources = _extract_center_tile(patterns_direct, int(args.tile_size))
            patterns_used = patterns_direct

        if not bool(args.no_color_match):
            patterns_used = _local_color_transfer(
                patterns_used,
                images,
                masks,
                ring_width=int(args.ring_width),
            )

        alpha_mask = _feather_mask(
            masks,
            kernel_size=int(args.feather_kernel),
            sigma=float(args.feather_sigma),
        )
        composites = _compose(images, patterns_used, alpha_mask)

    n = int(images.shape[0])

    rows: list[np.ndarray] = []
    for i in range(n):
        env_img = _annotate(_to_bgr_u8(images[i]), "Environment")
        mask_img = _annotate(_to_mask_vis(masks[i]), "Mask")
        pat_label = "Generated Pattern (Tiled)" if args.compose_mode == "tile" else "Generated Pattern"
        pat_img = _annotate(_to_bgr_u8(patterns_used[i]), pat_label)
        comp_img = _annotate(_to_bgr_u8(composites[i]), "Composite")

        row = np.concatenate([env_img, mask_img, pat_img, comp_img], axis=1)
        rows.append(row)

    comparison = np.concatenate(rows, axis=0)

    run_dir = Path(args.output_dir).resolve() / checkpoint_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = run_dir / args.output_comparison
    cv2.imwrite(str(comparison_path), comparison)

    # Requested flat 1x1 pattern image: source tile (not full canvas).
    pattern_flat_path = run_dir / args.output_pattern_flat
    first_tile = _to_bgr_u8(tile_sources[0])
    cv2.imwrite(str(pattern_flat_path), first_tile)

    # Also export all source tiles and used full-canvas patterns for convenience
    for i in range(n):
        tile_img = _to_bgr_u8(tile_sources[i])
        used_img = _to_bgr_u8(patterns_used[i])
        cv2.imwrite(str(run_dir / f"pattern_flat_1x1_sample{i+1:02d}.png"), tile_img)
        cv2.imwrite(str(run_dir / f"pattern_canvas_used_sample{i+1:02d}.png"), used_img)

    print(f"[visual inference] checkpoint={checkpoint_path.name}")
    print(
        f"[visual inference] split={args.split} samples={n} skipped={skipped} "
        f"device={device} compose_mode={args.compose_mode} tile_size={int(args.tile_size)} "
        f"random_tiling={(not bool(args.no_random_tiling))} color_match={(not bool(args.no_color_match))} "
        f"feather_kernel={int(args.feather_kernel)} feather_sigma={float(args.feather_sigma):.2f}"
    )
    print(f"[visual inference] comparison={comparison_path}")
    print(f"[visual inference] pattern_flat_1x1={pattern_flat_path}")


if __name__ == "__main__":
    main()
