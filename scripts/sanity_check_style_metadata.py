#!/usr/bin/env python3
"""Sanity checks for style metadata and style image bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate style metadata consistency")
    parser.add_argument("--metadata", default="camo/style_metadata/style_metadata.json", help="Path to style_metadata.json")
    parser.add_argument("--style-dir", default="camo/style_reference_clean", help="Directory with style images")
    args = parser.parse_args()

    metadata_path = Path(args.metadata).resolve()
    style_dir = Path(args.style_dir).resolve()

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not style_dir.exists():
        raise FileNotFoundError(f"Style directory not found: {style_dir}")

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    domains = payload.get("domains", {})
    if not isinstance(domains, dict) or not domains:
        raise RuntimeError("Invalid metadata: 'domains' must be a non-empty object")

    total_declared = 0
    for domain_name, entry in domains.items():
        files = entry.get("files", [])
        colors = entry.get("colors_rgb", [])
        freq = entry.get("frequency_profile", [])

        if not isinstance(files, list):
            raise RuntimeError(f"Invalid files list for domain: {domain_name}")
        if not isinstance(colors, list) or len(colors) == 0:
            raise RuntimeError(f"Empty palette colors for domain: {domain_name}")
        if not isinstance(freq, list) or len(freq) == 0:
            raise RuntimeError(f"Empty frequency profile for domain: {domain_name}")

        for c in colors:
            if not isinstance(c, list) or len(c) != 3:
                raise RuntimeError(f"Invalid RGB color entry in domain: {domain_name}")
            if any((not isinstance(x, (int, float)) or x < 0 or x > 255) for x in c):
                raise RuntimeError(f"Out-of-range RGB value in domain: {domain_name}")

        for file_name in files:
            p = style_dir / str(file_name)
            if not p.exists():
                raise RuntimeError(f"Missing file from metadata in domain '{domain_name}': {file_name}")

        total_declared += len(files)
        print(f"[style-sanity] {domain_name}: files={len(files)} colors={len(colors)} freq_bins={len(freq)}")

    print(f"[style-sanity] Total declared files: {total_declared}")
    print("[style-sanity] OK")


if __name__ == "__main__":
    main()
