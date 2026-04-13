from __future__ import annotations

from collections.abc import Sequence


def coerce_image_size_hw(raw_value: object, default: tuple[int, int] = (256, 256)) -> tuple[int, int]:
    """Normalize an image size config into (height, width)."""

    if raw_value is None:
        raw_value = default

    if isinstance(raw_value, int):
        side = int(raw_value)
        if side <= 0:
            raise ValueError(f"image_size must be > 0, got: {raw_value}")
        return side, side

    if isinstance(raw_value, str):
        value = raw_value.strip().lower().replace("×", "x")
        if "x" not in value:
            raise ValueError(f"image_size string must look like 'widthxheight', got: {raw_value!r}")
        left, right = [part.strip() for part in value.split("x", 1)]
        if not left or not right:
            raise ValueError(f"image_size string must look like 'widthxheight', got: {raw_value!r}")
        width = int(left)
        height = int(right)
        if width <= 0 or height <= 0:
            raise ValueError(f"image_size values must be > 0, got: {raw_value!r}")
        return height, width

    if isinstance(raw_value, Sequence):
        if len(raw_value) != 2:
            raise ValueError(f"image_size must have exactly 2 elements, got: {raw_value!r}")
        height = int(raw_value[0])
        width = int(raw_value[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"image_size values must be > 0, got: {raw_value!r}")
        return height, width

    raise TypeError(f"Unsupported image_size type: {type(raw_value).__name__}")


def format_image_size_wh(size_hw: tuple[int, int]) -> str:
    height, width = int(size_hw[0]), int(size_hw[1])
    return f"{width}x{height}"
