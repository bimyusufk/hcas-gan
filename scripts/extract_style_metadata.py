#!/usr/bin/env python3
"""Extract style metadata (domain mapping, palette, frequency profile) from style images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


CANONICAL_DOMAINS = ["woodland", "tropical_jungle", "arid_desert", "digital_camo"]


def infer_domain_from_name(file_name: str) -> str:
    n = file_name.lower()
    if "digital" in n or "marpat" in n or "cadpat" in n or "ucp" in n:
        return "digital_camo"
    if "desert" in n or "arid" in n or "dpm oman" in n or "dpm yemen" in n:
        return "arid_desert"
    if "jungle" in n or "tropical" in n or "brushstroke" in n:
        return "tropical_jungle"
    return "woodland"


def estimate_palette(image_paths: list[Path], n_colors: int, max_pixels_total: int = 800_000) -> list[list[int]]:
    if not image_paths:
        return []

    samples = []
    budget_per_image = max(2000, max_pixels_total // max(len(image_paths), 1))

    for p in image_paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        flat = rgb.reshape(-1, 3).astype(np.float32)
        if flat.shape[0] > budget_per_image:
            idx = np.random.choice(flat.shape[0], size=budget_per_image, replace=False)
            flat = flat[idx]
        samples.append(flat)

    if not samples:
        return []

    data = np.concatenate(samples, axis=0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.2)
    _, _, centers = cv2.kmeans(data, n_colors, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    centers = np.clip(np.round(centers), 0, 255).astype(np.int32)
    return centers.tolist()


def radial_frequency_profile(image_paths: list[Path], image_size: int = 256, n_bins: int = 32) -> list[float]:
    profiles = []

    for p in image_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
        x = img.astype(np.float32) / 255.0

        fft = np.fft.fftshift(np.fft.fft2(x))
        power = np.abs(fft) ** 2

        h, w = power.shape
        yy, xx = np.indices((h, w))
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        rr_bin = np.clip((rr / (rr.max() + 1e-8) * (n_bins - 1)).astype(np.int32), 0, n_bins - 1)

        sums = np.bincount(rr_bin.reshape(-1), weights=power.reshape(-1), minlength=n_bins).astype(np.float32)
        counts = np.bincount(rr_bin.reshape(-1), minlength=n_bins).astype(np.float32)
        profile = sums / np.clip(counts, 1.0, None)
        profile = profile / max(float(profile.mean()), 1e-8)
        profiles.append(profile)

    if not profiles:
        return [1.0] * n_bins

    return np.mean(np.stack(profiles, axis=0), axis=0).astype(np.float32).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract style metadata for training")
    parser.add_argument("--input-dir", default="camo/style_reference_clean", help="Directory with style images")
    parser.add_argument("--output-dir", default="camo/style_metadata", help="Output metadata directory")
    parser.add_argument("--n-colors", type=int, default=6, help="Palette size (k-means)")
    parser.add_argument("--freq-bins", type=int, default=32, help="Number of radial frequency bins")
    parser.add_argument("--image-size", type=int, default=256, help="Resize for frequency analysis")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([p for p in input_dir.glob("*.png") if p.is_file()])
    if not image_files:
        raise RuntimeError(f"No PNG files found in {input_dir}")

    grouped: dict[str, list[Path]] = {k: [] for k in CANONICAL_DOMAINS}
    for p in image_files:
        domain = infer_domain_from_name(p.name)
        grouped[domain].append(p)

    metadata = {
        "source_dir": str(input_dir),
        "total_files": len(image_files),
        "domains": {},
    }

    mapping_rows = []
    for domain in CANONICAL_DOMAINS:
        files = grouped[domain]
        rel_names = [f.name for f in files]
        colors = estimate_palette(files, n_colors=int(args.n_colors))
        freq_profile = radial_frequency_profile(
            files,
            image_size=int(args.image_size),
            n_bins=int(args.freq_bins),
        )

        metadata["domains"][domain] = {
            "count": len(files),
            "files": rel_names,
            "colors_rgb": colors,
            "frequency_profile": freq_profile,
            "n_colors": int(args.n_colors),
            "freq_bins": int(args.freq_bins),
        }

        for name in rel_names:
            mapping_rows.append({"file": name, "domain": domain})

    metadata_path = output_dir / "style_metadata_.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    mapping_path = output_dir / "style_domain_map.csv"
    with mapping_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "domain"])
        writer.writeheader()
        writer.writerows(mapping_rows)

    print(f"[style_metadata] Total files: {len(image_files)}")
    for domain in CANONICAL_DOMAINS:
        print(f"[style_metadata] {domain}: {len(grouped[domain])}")
    print(f"[style_metadata] Saved: {metadata_path}")
    print(f"[style_metadata] Saved: {mapping_path}")


if __name__ == "__main__":
    main()
