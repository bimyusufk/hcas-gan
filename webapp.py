"""Simple web app for HCAS-GAN visual inference.

Features:
1) Upload environment image.
2) Annotate camou polygon on canvas.
3) Run generator inference from checkpoint.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from flask import Flask, jsonify, render_template, request, send_file


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import GeneratorUNet


CHECKPOINT_NAME_RE = re.compile(r"checkpoint_epoch_(\d+)_step_(\d+)(?:_[^.]+)?\.pt$")
CHECKPOINT_VARIANT_RE = re.compile(r"(?:^|[_\-])v(\d+)(?:$|[_\-.])", re.IGNORECASE)


def _read_json_relaxed(json_path: Path) -> dict[str, Any]:
    raw = json_path.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
        return json.loads(cleaned)


def _normalize_label(label: str) -> str:
    return str(label).strip().lower()


def _resolve_image_path(
    annotation_path: Path,
    image_path_field: str | None,
    image_suffix: str,
) -> Path:
    candidates: list[Path] = []
    if image_path_field:
        field_as_path = Path(str(image_path_field))
        candidates.append((annotation_path.parent / field_as_path).resolve())
        candidates.append((annotation_path.parent / field_as_path.name).resolve())

    candidates.append(annotation_path.with_suffix(image_suffix).resolve())
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Image pair not found for annotation: {annotation_path.name}")


def _resolve_device(policy: str) -> torch.device:
    p = str(policy).strip().lower()
    if p == "cpu":
        return torch.device("cpu")
    if p == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if p == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_base64_png(img_bgr_u8: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", img_bgr_u8)
    if not ok:
        raise RuntimeError("Failed to encode image to PNG.")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def _build_mask(h: int, w: int, points: list[list[float]]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(points) < 3:
        return mask

    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return mask

    pts = np.round(pts).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    cv2.fillPoly(mask, [pts], color=1)
    return mask


def _parse_checkpoint_name(checkpoint_path: Path) -> tuple[int | None, int | None]:
    match = CHECKPOINT_NAME_RE.match(checkpoint_path.name)
    if match is None:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _checkpoint_sort_key(checkpoint_path: Path) -> tuple[int, int, str]:
    epoch, step = _parse_checkpoint_name(checkpoint_path)
    epoch_key = epoch if epoch is not None else 10**9
    step_key = step if step is not None else 10**9
    return (epoch_key, step_key, checkpoint_path.name)


def _format_checkpoint_label(checkpoint_path: Path) -> str:
    epoch, step = _parse_checkpoint_name(checkpoint_path)
    if epoch is None or step is None:
        return checkpoint_path.name
    return f"epoch {epoch} | step {step}"


def _display_checkpoint_path(checkpoint_path: Path) -> str:
    try:
        return str(checkpoint_path.resolve().relative_to(ROOT))
    except Exception:
        return str(checkpoint_path)


def _infer_checkpoint_variant(checkpoint_path: Path) -> int | None:
    name = checkpoint_path.name
    match = CHECKPOINT_VARIANT_RE.search(name)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


class InferenceService:
    def __init__(
        self,
        checkpoint_path: Path,
        config_path: Path,
        checkpoint_dir: Path | None = None,
        device_policy: str = "auto",
        compose_mode: str = "tile",
        tile_size: int = 64,
        random_tiling: bool = True,
        color_match: bool = True,
        ring_width: int = 11,
        feather_kernel: int = 15,
        feather_sigma: float = 3.0,
        input_pad: int = 16,
        edge_crop: int = 8,
    ):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path.resolve()
        self.checkpoint_dir = (checkpoint_dir or self.checkpoint_path.parent).resolve()

        config = self._load_config(config_path)
        data_cfg = config.get("data", {})
        annotations_dir_raw = str(data_cfg.get("annotations_dir", "./camo/environment_annotation"))
        ann_dir = Path(annotations_dir_raw)
        if not ann_dir.is_absolute():
            ann_dir = (ROOT / ann_dir).resolve()

        self.annotations_dir = ann_dir
        self.image_suffix = str(data_cfg.get("image_suffix", ".png"))
        self.annotation_suffix = str(data_cfg.get("annotation_suffix", ".json"))
        self.target_label = _normalize_label(str(data_cfg.get("target_label", "camou")))
        self.ignore_labels = {
            _normalize_label(str(label))
            for label in data_cfg.get("ignore_labels", ["skin"])
        }
        self.min_polygon_points = 3

        self.image_size = int(config.get("hcas_specific", {}).get("image_size", 256))
        self.compose_mode = str(compose_mode).strip().lower()
        self.tile_size = int(max(4, tile_size))
        self.random_tiling = bool(random_tiling)
        self.color_match = bool(color_match)
        self.ring_width = int(max(1, ring_width))
        self.feather_kernel = int(max(1, feather_kernel))
        self.feather_sigma = float(max(0.1, feather_sigma))
        self.input_pad = int(max(0, input_pad))
        self.edge_crop = int(max(0, edge_crop))

        self.default_device_policy = str(device_policy).strip().lower() or "auto"
        self.default_device = _resolve_device(self.default_device_policy)

        self._generator_cache: dict[str, GeneratorUNet] = {}
        self.available_checkpoint_paths = self._load_available_checkpoint_paths(self.checkpoint_dir, self.checkpoint_path)
        self.available_config_paths = self._load_available_config_paths(self.config_path)
        self.default_checkpoint_config_map = self._build_default_checkpoint_config_map()
        self.environment_samples = self._load_environment_samples()
        self.environment_sample_map = {sample["sample_id"]: sample for sample in self.environment_samples}

        # Preload default checkpoint into cache (CPU by default; moved per-request if needed).
        default_config = self.default_checkpoint_config_map.get(self.checkpoint_path.name, self.config_path)
        self._get_generator_for_checkpoint(self.checkpoint_path, default_config)

    @staticmethod
    def _load_available_checkpoint_paths(checkpoint_dir: Path, primary_checkpoint: Path) -> tuple[Path, ...]:
        paths: list[Path] = []
        if checkpoint_dir.exists():
            paths.extend(sorted(checkpoint_dir.glob("checkpoint_*.pt"), key=_checkpoint_sort_key))

        if primary_checkpoint.exists():
            paths.append(primary_checkpoint.resolve())

        unique_paths: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            resolved = path.resolve()
            key = str(resolved)
            if key in seen or not resolved.exists():
                continue
            unique_paths.append(resolved)
            seen.add(key)

        return tuple(sorted(unique_paths, key=_checkpoint_sort_key))

    @staticmethod
    def _load_available_config_paths(primary_config: Path) -> tuple[Path, ...]:
        paths: list[Path] = []
        paths.extend(sorted(ROOT.glob("config*.yaml")))
        paths.extend(sorted(ROOT.glob("config*.yml")))
        if primary_config.exists():
            paths.append(primary_config.resolve())

        unique_paths: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            resolved = path.resolve()
            key = str(resolved)
            if key in seen or not resolved.exists() or not resolved.is_file():
                continue
            unique_paths.append(resolved)
            seen.add(key)

        return tuple(sorted(unique_paths, key=lambda p: p.name.lower()))

    def _infer_default_config_for_checkpoint(self, checkpoint_path: Path) -> Path:
        variant = _infer_checkpoint_variant(checkpoint_path)

        preferred_names: list[str] = []
        if variant == 1:
            preferred_names.extend(["config.run_a.yaml", "config_run_a.yaml", "config.yaml"])
        elif variant == 2:
            preferred_names.extend(["config.run_b.yaml", "config_run_b.yaml", "config.yaml"])

        preferred_names.append(self.config_path.name)

        available_by_name: dict[str, Path] = {path.name.lower(): path for path in self.available_config_paths}
        for candidate_name in preferred_names:
            match = available_by_name.get(str(candidate_name).strip().lower())
            if match is not None:
                return match

        if self.available_config_paths:
            return self.available_config_paths[0]
        return self.config_path

    def _build_default_checkpoint_config_map(self) -> dict[str, Path]:
        mapping: dict[str, Path] = {}
        for checkpoint_path in self.available_checkpoint_paths:
            mapping[checkpoint_path.name] = self._infer_default_config_for_checkpoint(checkpoint_path)
        if self.checkpoint_path.name not in mapping:
            mapping[self.checkpoint_path.name] = self._infer_default_config_for_checkpoint(self.checkpoint_path)
        return mapping

    def _extract_annotation_options(self, annotation_data: dict[str, Any]) -> list[dict[str, Any]]:
        shapes = annotation_data.get("shapes", [])
        if not isinstance(shapes, list):
            return []

        options: list[dict[str, Any]] = []
        option_index = 0
        for shape in shapes:
            if not isinstance(shape, dict):
                continue

            label_raw = str(shape.get("label", ""))
            label = _normalize_label(label_raw)
            if label in self.ignore_labels:
                continue
            if label != self.target_label:
                continue
            if str(shape.get("shape_type", "polygon")).lower() != "polygon":
                continue

            points = shape.get("points", [])
            if not isinstance(points, list):
                continue

            clean_points: list[list[float]] = []
            for point in points:
                if not isinstance(point, list) or len(point) < 2:
                    continue
                try:
                    x = float(point[0])
                    y = float(point[1])
                except Exception:
                    continue
                clean_points.append([x, y])

            if len(clean_points) < self.min_polygon_points:
                continue

            options.append(
                {
                    "annotation_id": f"ann_{option_index:03d}",
                    "label": label_raw or self.target_label,
                    "point_count": len(clean_points),
                    "points": clean_points,
                }
            )
            option_index += 1

        return options

    def _load_environment_samples(self) -> list[dict[str, Any]]:
        if not self.annotations_dir.exists():
            return []

        samples: list[dict[str, Any]] = []
        ann_files = sorted(self.annotations_dir.glob(f"*{self.annotation_suffix}"))
        for ann_path in ann_files:
            try:
                annotation_data = _read_json_relaxed(ann_path)
                image_path = _resolve_image_path(
                    ann_path,
                    image_path_field=annotation_data.get("imagePath"),
                    image_suffix=self.image_suffix,
                )
                annotation_options = self._extract_annotation_options(annotation_data)
                if not annotation_options:
                    continue

                sample_id = ann_path.name
                samples.append(
                    {
                        "sample_id": sample_id,
                        "image_name": image_path.name,
                        "annotation_file": ann_path.name,
                        "image_path": image_path,
                        "annotation_path": ann_path,
                        "annotation_options": annotation_options,
                    }
                )
            except Exception:
                continue

        samples.sort(key=lambda sample: (sample.get("image_name", ""), sample.get("sample_id", "")))
        return samples

    def get_environment_samples(self) -> list[dict[str, Any]]:
        return [
            {
                "sample_id": sample["sample_id"],
                "image_name": sample["image_name"],
                "annotation_file": sample["annotation_file"],
                "annotation_count": len(sample["annotation_options"]),
            }
            for sample in self.environment_samples
        ]

    def get_annotations_for_sample(self, sample_id: str) -> list[dict[str, Any]]:
        sample = self.environment_sample_map.get(str(sample_id).strip())
        if sample is None:
            return []
        return [
            {
                "annotation_id": ann["annotation_id"],
                "label": ann["label"],
                "point_count": ann["point_count"],
            }
            for ann in sample["annotation_options"]
        ]

    def get_sample_image_path(self, sample_id: str) -> Path:
        sample = self.environment_sample_map.get(str(sample_id).strip())
        if sample is None:
            raise KeyError(f"Unknown sample_id: {sample_id}")
        return Path(sample["image_path"])

    def get_points_for_annotation(self, sample_id: str, annotation_id: str) -> list[list[float]]:
        sample = self.environment_sample_map.get(str(sample_id).strip())
        if sample is None:
            raise KeyError(f"Unknown sample_id: {sample_id}")

        ann_id_norm = str(annotation_id).strip()
        for ann in sample["annotation_options"]:
            if str(ann.get("annotation_id", "")).strip() == ann_id_norm:
                return list(ann["points"])

        raise KeyError(f"Unknown annotation_id '{annotation_id}' for sample '{sample_id}'")

    def get_checkpoint_options(self) -> list[dict[str, Any]]:
        options: list[dict[str, Any]] = []
        default_path = self.checkpoint_path.resolve()
        for path in self.available_checkpoint_paths:
            epoch, step = _parse_checkpoint_name(path)
            default_config = self.default_checkpoint_config_map.get(path.name, self.config_path)
            options.append(
                {
                    "name": path.name,
                    "path": _display_checkpoint_path(path),
                    "label": _format_checkpoint_label(path),
                    "epoch": epoch,
                    "step": step,
                    "is_default": path.resolve() == default_path,
                    "default_config_name": default_config.name,
                }
            )
        return options

    def get_model_config_options(self) -> list[dict[str, Any]]:
        return [
            {
                "name": path.name,
                "path": _display_checkpoint_path(path),
                "is_default": path.resolve() == self.config_path.resolve(),
            }
            for path in self.available_config_paths
        ]

    def get_default_selected_checkpoint_names(self, count: int = 2) -> list[str]:
        available = list(self.available_checkpoint_paths)
        if not available:
            return [self.checkpoint_path.name]

        max_count = max(1, int(count))
        names: list[str] = []
        primary_name = self.checkpoint_path.name
        if any(path.name == primary_name for path in available):
            names.append(primary_name)

        for path in reversed(available):
            if path.name == primary_name:
                continue
            names.append(path.name)
            if len(names) >= max_count:
                break

        if not names:
            names = [available[-1].name]

        return names[:max_count]

    def _get_generator_for_checkpoint(self, checkpoint_path: Path, model_config_path: Path) -> GeneratorUNet:
        resolved = checkpoint_path.resolve()
        resolved_cfg = model_config_path.resolve()
        cache_key = f"{resolved}::{resolved_cfg}"
        generator = self._generator_cache.get(cache_key)
        if generator is None:
            generator = self._load_generator(resolved, torch.device("cpu"), config_path=resolved_cfg)
            self._generator_cache[cache_key] = generator
        return generator

    @staticmethod
    def _extract_center_tile(pattern_t: torch.Tensor, tile_size: int) -> torch.Tensor:
        _, _, h, w = pattern_t.shape
        t = max(4, min(int(tile_size), int(h), int(w)))
        y0 = (int(h) - t) // 2
        x0 = (int(w) - t) // 2
        return pattern_t[:, :, y0 : y0 + t, x0 : x0 + t]

    @staticmethod
    def _reflect_pad_nchw(x: torch.Tensor, pad: int) -> torch.Tensor:
        p = max(0, int(pad))
        if p <= 0:
            return x
        h, w = int(x.shape[2]), int(x.shape[3])
        max_p = max(0, (min(h, w) // 2) - 1)
        p = min(p, max_p)
        if p <= 0:
            return x
        return F.pad(x, (p, p, p, p), mode="reflect")

    @staticmethod
    def _remove_nchw_pad(x: torch.Tensor, pad: int) -> torch.Tensor:
        p = max(0, int(pad))
        if p <= 0:
            return x
        h, w = int(x.shape[2]), int(x.shape[3])
        if (h - (2 * p)) < 2 or (w - (2 * p)) < 2:
            return x
        return x[:, :, p : h - p, p : w - p]

    @staticmethod
    def _edge_crop_and_resize_nchw(x: torch.Tensor, crop: int) -> torch.Tensor:
        c = max(0, int(crop))
        if c <= 0:
            return x
        h, w = int(x.shape[2]), int(x.shape[3])
        if (h - (2 * c)) < 8 or (w - (2 * c)) < 8:
            return x
        cropped = x[:, :, c : h - c, c : w - c]
        if cropped.shape[2:] == x.shape[2:]:
            return cropped
        return F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)

    @staticmethod
    def _tile_to_canvas(
        tile_t: torch.Tensor,
        out_h: int,
        out_w: int,
        *,
        randomize: bool,
        seed: int,
    ) -> torch.Tensor:
        n, c, t_h, t_w = tile_t.shape
        canvas = torch.zeros((n, c, out_h, out_w), dtype=tile_t.dtype, device=tile_t.device)
        for bi in range(n):
            rng = np.random.default_rng(seed + bi)
            y = 0
            while y < out_h:
                x = 0
                while x < out_w:
                    patch = tile_t[bi]
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

    @staticmethod
    def _ensure_odd(v: int) -> int:
        x = max(1, int(v))
        return x if (x % 2 == 1) else (x + 1)

    @staticmethod
    def _feather_mask(mask_t: torch.Tensor, *, kernel_size: int, sigma: float) -> torch.Tensor:
        k = InferenceService._ensure_odd(kernel_size)
        s = max(0.1, float(sigma))
        out_np = []
        for i in range(mask_t.shape[0]):
            m = mask_t[i, 0].detach().float().cpu().clamp(0.0, 1.0).numpy()
            if k > 1:
                m = cv2.GaussianBlur(m, (k, k), sigmaX=s, sigmaY=s)
            m = np.clip(m, 0.0, 1.0)
            out_np.append(m)
        out = torch.from_numpy(np.stack(out_np, axis=0)).unsqueeze(1).to(mask_t.device, dtype=mask_t.dtype)
        return out

    @staticmethod
    def _local_color_transfer(
        pattern_t: torch.Tensor,
        background_t: torch.Tensor,
        mask_t: torch.Tensor,
        *,
        ring_width: int,
    ) -> torch.Tensor:
        rw = max(1, int(ring_width))
        kernel = np.ones((rw, rw), dtype=np.uint8)
        out_samples: list[torch.Tensor] = []
        for bi in range(pattern_t.shape[0]):
            p = pattern_t[bi].detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
            b = background_t[bi].detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
            m = (mask_t[bi, 0].detach().float().cpu().numpy() > 0.5).astype(np.uint8)
            if int(m.sum()) == 0:
                out_samples.append(pattern_t[bi].detach().cpu())
                continue

            dil = cv2.dilate(m, kernel, iterations=1)
            ring = (dil > 0) & (m == 0)
            if int(ring.sum()) < 32:
                ring = np.ones_like(m, dtype=bool)

            m_bool = m > 0
            p_sel = p[m_bool]
            b_sel = b[ring]
            if p_sel.size == 0 or b_sel.size == 0:
                out_samples.append(pattern_t[bi].detach().cpu())
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

        out_batch = torch.stack(out_samples, dim=0).to(pattern_t.device, dtype=pattern_t.dtype)
        return out_batch

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _load_generator(
        checkpoint_path: Path,
        device: torch.device,
        *,
        config_path: Path | None = None,
    ) -> GeneratorUNet:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        gen_state = data.get("generator_state")
        if not isinstance(gen_state, dict):
            raise RuntimeError("Checkpoint does not contain generator_state")

        gen_kwargs: dict[str, Any] = {
            "in_channels": 3,
            "out_channels": 3,
            "base_channels": 64,
            "output_activation": "sigmoid",
        }
        if config_path is not None and config_path.exists():
            cfg = InferenceService._load_config(config_path)
            gen_cfg = cfg.get("model", {}).get("generator", {})
            if isinstance(gen_cfg, dict):
                try:
                    gen_kwargs["in_channels"] = int(gen_cfg.get("in_channels", gen_kwargs["in_channels"]))
                    gen_kwargs["out_channels"] = int(gen_cfg.get("out_channels", gen_kwargs["out_channels"]))
                    gen_kwargs["base_channels"] = int(gen_cfg.get("base_channels", gen_kwargs["base_channels"]))
                    gen_kwargs["output_activation"] = str(
                        gen_cfg.get("output_activation", gen_kwargs["output_activation"])
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Invalid generator config in '{config_path.name}': {type(exc).__name__}: {exc}"
                    ) from exc

        model = GeneratorUNet(**gen_kwargs).to(device)
        model.load_state_dict(gen_state, strict=True)
        model.eval()
        return model

    def _resolve_runtime_device(self, device_policy: str | None) -> torch.device:
        policy = str(device_policy or self.default_device_policy).strip().lower()
        if policy not in {"auto", "cpu", "cuda"}:
            policy = self.default_device_policy
        return _resolve_device(policy)

    def _infer_single_checkpoint(
        self,
        checkpoint_path: Path,
        model_config_path: Path,
        image_t: torch.Tensor,
        mask_t: torch.Tensor,
        *,
        runtime_device: torch.device,
        compose_mode: str,
        tile_size: int,
    ) -> dict[str, Any]:
        generator = self._get_generator_for_checkpoint(checkpoint_path, model_config_path)
        moved_to_device = False
        if runtime_device.type == "cuda":
            generator.to(runtime_device)
            moved_to_device = True
        else:
            generator.to("cpu")

        try:
            with torch.no_grad():
                image_for_gen = self._reflect_pad_nchw(image_t, self.input_pad)
                pattern_padded_t = generator(image_for_gen)
                pattern_direct_t = self._remove_nchw_pad(pattern_padded_t, self.input_pad)
                pattern_direct_t = self._edge_crop_and_resize_nchw(pattern_direct_t, self.edge_crop)
                tile_src_t = self._extract_center_tile(pattern_direct_t, tile_size)
                if compose_mode == "tile":
                    pattern_used_t = self._tile_to_canvas(
                        tile_src_t,
                        int(self.image_size),
                        int(self.image_size),
                        randomize=self.random_tiling,
                        seed=1234,
                    )
                else:
                    pattern_used_t = pattern_direct_t

                if self.color_match:
                    pattern_used_t = self._local_color_transfer(
                        pattern_used_t,
                        image_t,
                        mask_t,
                        ring_width=self.ring_width,
                    )

                alpha_t = self._feather_mask(
                    mask_t,
                    kernel_size=self.feather_kernel,
                    sigma=self.feather_sigma,
                )
                composite_t = (image_t * (1.0 - alpha_t)) + (pattern_used_t * alpha_t)

            pat_u8 = (
                tile_src_t[0].detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0
            ).round().astype(np.uint8)
            pat_canvas_u8 = (
                pattern_used_t[0].detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0
            ).round().astype(np.uint8)
            comp_u8 = (
                composite_t[0].detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0
            ).round().astype(np.uint8)

            pat_bgr = cv2.cvtColor(pat_u8, cv2.COLOR_RGB2BGR)
            pat_canvas_bgr = cv2.cvtColor(pat_canvas_u8, cv2.COLOR_RGB2BGR)
            comp_bgr = cv2.cvtColor(comp_u8, cv2.COLOR_RGB2BGR)

            return {
                "checkpoint_name": checkpoint_path.name,
                "checkpoint_path": _display_checkpoint_path(checkpoint_path),
                "checkpoint_label": _format_checkpoint_label(checkpoint_path),
                "model_config_name": model_config_path.name,
                "model_config_path": _display_checkpoint_path(model_config_path),
                "pattern": _to_base64_png(pat_bgr),
                "pattern_canvas": _to_base64_png(pat_canvas_bgr),
                "composite": _to_base64_png(comp_bgr),
                "compose_mode": compose_mode,
                "tile_size": int(tile_size),
                "random_tiling": bool(self.random_tiling),
                "color_match": bool(self.color_match),
                "feather_kernel": int(self.feather_kernel),
                "feather_sigma": float(self.feather_sigma),
                "input_pad": int(self.input_pad),
                "edge_crop": int(self.edge_crop),
                "runtime_device": str(runtime_device),
            }
        finally:
            if moved_to_device:
                generator.to("cpu")

    def infer(
        self,
        image_bgr: np.ndarray,
        polygon_points: list[list[float]],
        *,
        checkpoint_paths: list[Path] | None = None,
        checkpoint_config_paths: dict[str, Path] | None = None,
        device_policy: str | None = None,
        compose_mode: str | None = None,
        tile_size: int | None = None,
    ) -> dict[str, Any]:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Input image must be HxWx3.")

        h0, w0 = image_bgr.shape[:2]
        mask0 = _build_mask(h0, w0, polygon_points)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask0, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)

        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()

        runtime_device = self._resolve_runtime_device(device_policy)

        image_t = image_t.unsqueeze(0).to(runtime_device)
        mask_t = mask_t.unsqueeze(0).to(runtime_device)

        mode = (compose_mode or self.compose_mode).strip().lower()
        if mode not in {"direct", "tile"}:
            mode = self.compose_mode
        tsize = int(max(4, tile_size if tile_size is not None else self.tile_size))

        selected_paths = list(checkpoint_paths or [self.checkpoint_path])
        unique_paths: list[Path] = []
        seen_paths: set[str] = set()
        for path in selected_paths:
            resolved = path.resolve()
            key = str(resolved)
            if key in seen_paths:
                continue
            if not resolved.exists():
                raise FileNotFoundError(f"Checkpoint not found: {resolved}")
            unique_paths.append(resolved)
            seen_paths.add(key)

        if not unique_paths:
            unique_paths = [self.checkpoint_path]

        config_overrides = {
            str(name).strip(): path.resolve()
            for name, path in (checkpoint_config_paths or {}).items()
            if str(name).strip()
        }
        config_by_checkpoint_key: dict[str, Path] = {}
        for path in unique_paths:
            ck_key = str(path.resolve())
            selected_cfg = config_overrides.get(path.name)
            if selected_cfg is None:
                selected_cfg = self.default_checkpoint_config_map.get(path.name, self.config_path)
            selected_cfg = selected_cfg.resolve()
            if not selected_cfg.exists() or not selected_cfg.is_file():
                raise FileNotFoundError(f"Config not found for checkpoint '{path.name}': {selected_cfg}")
            config_by_checkpoint_key[ck_key] = selected_cfg

        checkpoint_summaries = [
            {
                "name": path.name,
                "path": _display_checkpoint_path(path),
                "label": _format_checkpoint_label(path),
                "is_default": path.resolve() == self.checkpoint_path.resolve(),
                "selected_config_name": config_by_checkpoint_key[str(path.resolve())].name,
                "selected_config_path": _display_checkpoint_path(config_by_checkpoint_key[str(path.resolve())]),
            }
            for path in unique_paths
        ]

        env_u8 = (image_t[0].detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        mask_u8 = (mask_t[0, 0].detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)

        env_bgr = cv2.cvtColor(env_u8, cv2.COLOR_RGB2BGR)
        mask_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)

        results = [
            self._infer_single_checkpoint(
                path,
                config_by_checkpoint_key[str(path.resolve())],
                image_t,
                mask_t,
                runtime_device=runtime_device,
                compose_mode=mode,
                tile_size=tsize,
            )
            for path in unique_paths
        ]

        return {
            "environment": _to_base64_png(env_bgr),
            "mask": _to_base64_png(mask_bgr),
            "width": int(self.image_size),
            "height": int(self.image_size),
            "mask_area_ratio": float(mask.mean()),
            "selected_checkpoints": checkpoint_summaries,
            "runtime_device": str(runtime_device),
            "results": results,
        }

    def infer_from_environment_selection(
        self,
        *,
        sample_id: str,
        annotation_id: str,
        checkpoint_paths: list[Path] | None = None,
        checkpoint_config_paths: dict[str, Path] | None = None,
        device_policy: str | None = None,
        compose_mode: str | None = None,
        tile_size: int | None = None,
    ) -> dict[str, Any]:
        image_path = self.get_sample_image_path(sample_id)
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read selected image: {image_path}")

        points = self.get_points_for_annotation(sample_id, annotation_id)
        result = self.infer(
            image_bgr,
            points,
            checkpoint_paths=checkpoint_paths,
            checkpoint_config_paths=checkpoint_config_paths,
            device_policy=device_policy,
            compose_mode=compose_mode,
            tile_size=tile_size,
        )
        result["sample_id"] = str(sample_id)
        result["annotation_id"] = str(annotation_id)
        result["image_name"] = image_path.name
        return result


def _find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    candidates = sorted(checkpoint_dir.glob("checkpoint_*.pt"), key=_checkpoint_sort_key)
    return candidates[-1] if candidates else None


def _resolve_checkpoint_arg(raw_checkpoint: str) -> Path:
    """Resolve checkpoint from absolute path, relative path, or bare filename.

    Resolution order:
    1) absolute path as-is
    2) ROOT / raw_checkpoint
    3) ROOT / "checkpoints" / raw_checkpoint
    """
    candidate = Path(raw_checkpoint)
    if candidate.is_absolute():
        return candidate

    direct = (ROOT / candidate).resolve()
    if direct.exists():
        return direct

    in_checkpoints = (ROOT / "checkpoints" / candidate.name).resolve()
    if in_checkpoints.exists():
        return in_checkpoints

    return direct


def _resolve_config_arg(raw_config: str, default_config: Path | None = None) -> Path:
    """Resolve config file from absolute path, relative path, or bare filename."""
    candidate = Path(str(raw_config).strip())
    if candidate.is_absolute():
        return candidate

    direct = (ROOT / candidate).resolve()
    if direct.exists():
        return direct

    in_root = (ROOT / candidate.name).resolve()
    if in_root.exists():
        return in_root

    if default_config is not None:
        return default_config.resolve()
    return direct


def create_app(service: InferenceService) -> Flask:
    app = Flask(__name__, template_folder=str(ROOT / "web" / "templates"), static_folder=str(ROOT / "web" / "static"))

    model_glb_path = ROOT / "long_sleeve_t-_shirt.glb"

    @app.get("/")
    def index() -> str:
        return render_template(
            "index.html",
            checkpoint_name=service.checkpoint_path.name,
            image_size=service.image_size,
            default_device_policy=service.default_device_policy,
            compose_mode=service.compose_mode,
            tile_size=service.tile_size,
            model_glb_url="/model/long_sleeve_t-_shirt.glb",
        )

    @app.get("/api/environment/samples")
    def list_environment_samples() -> Any:
        samples = service.get_environment_samples()
        default_sample_id = samples[0]["sample_id"] if samples else ""
        return jsonify(
            {
                "samples": samples,
                "default_sample_id": default_sample_id,
            }
        )

    @app.get("/api/environment/annotations")
    def list_sample_annotations() -> tuple[Any, int] | Any:
        sample_id = str(request.args.get("sample_id", "")).strip()
        if not sample_id:
            return jsonify({"error": "Missing sample_id query parameter."}), 400

        annotations = service.get_annotations_for_sample(sample_id)
        if not annotations:
            return jsonify({"error": f"No annotations found for sample_id={sample_id}"}), 404

        return jsonify(
            {
                "sample_id": sample_id,
                "annotations": annotations,
                "default_annotation_id": annotations[0]["annotation_id"],
            }
        )

    @app.get("/api/environment/image")
    def get_environment_image() -> tuple[Any, int] | Any:
        sample_id = str(request.args.get("sample_id", "")).strip()
        if not sample_id:
            return jsonify({"error": "Missing sample_id query parameter."}), 400

        try:
            image_path = service.get_sample_image_path(sample_id)
        except Exception as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 404

        return send_file(image_path, mimetype="image/png")

    @app.get("/api/checkpoints")
    def list_checkpoints() -> Any:
        return jsonify(
            {
                "available_checkpoints": service.get_checkpoint_options(),
                "available_configs": service.get_model_config_options(),
                "default_checkpoint_config_map": {
                    name: path.name for name, path in service.default_checkpoint_config_map.items()
                },
                "default_selected_checkpoints": service.get_default_selected_checkpoint_names(),
                "default_checkpoint": {
                    "name": service.checkpoint_path.name,
                    "path": _display_checkpoint_path(service.checkpoint_path),
                    "label": _format_checkpoint_label(service.checkpoint_path),
                },
            }
        )

    @app.get("/model/long_sleeve_t-_shirt.glb")
    def model_glb() -> Any:
        if not model_glb_path.exists():
            return jsonify({"error": f"Model file not found: {model_glb_path.name}"}), 404
        return send_file(model_glb_path, mimetype="model/gltf-binary")

    @app.post("/api/infer")
    def infer() -> tuple[Any, int] | Any:
        payload = request.get_json(silent=True) or {}

        sample_id = str(payload.get("sample_id", "")).strip()
        annotation_id = str(payload.get("annotation_id", "")).strip()
        if not sample_id:
            return jsonify({"error": "Missing sample_id in request body."}), 400
        if not annotation_id:
            return jsonify({"error": "Missing annotation_id in request body."}), 400

        selected_checkpoint_names = payload.get("selected_checkpoints", [])
        if not isinstance(selected_checkpoint_names, list):
            return jsonify({"error": "selected_checkpoints must be a list."}), 400

        checkpoint_config_map_raw = payload.get("checkpoint_config_map", {})
        if checkpoint_config_map_raw is None:
            checkpoint_config_map_raw = {}
        if not isinstance(checkpoint_config_map_raw, dict):
            return jsonify({"error": "checkpoint_config_map must be an object/dict."}), 400

        checkpoint_paths = [_resolve_checkpoint_arg(str(name)) for name in selected_checkpoint_names if str(name).strip()]
        checkpoint_config_paths: dict[str, Path] = {}
        for checkpoint_name, config_name in checkpoint_config_map_raw.items():
            ck_name = str(checkpoint_name).strip()
            cfg_name = str(config_name).strip()
            if not ck_name or not cfg_name:
                continue
            checkpoint_config_paths[ck_name] = _resolve_config_arg(cfg_name, service.config_path)

        device_policy = str(payload.get("device", service.default_device_policy)).strip().lower()
        compose_mode = str(payload.get("compose_mode", service.compose_mode)).strip().lower()
        if compose_mode not in {"tile", "direct"}:
            compose_mode = service.compose_mode

        try:
            tile_size = int(payload.get("tile_size", service.tile_size))
        except Exception:
            tile_size = int(service.tile_size)

        try:
            result = service.infer_from_environment_selection(
                sample_id=sample_id,
                annotation_id=annotation_id,
                checkpoint_paths=checkpoint_paths,
                checkpoint_config_paths=checkpoint_config_paths,
                device_policy=device_policy,
                compose_mode=compose_mode,
                tile_size=tile_size,
            )
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": f"Inference failed: {type(exc).__name__}: {exc}"}), 500

    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HCAS-GAN web inference app")
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint path. If empty, latest in checkpoints/ is used.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config YAML path")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Runtime device")
    parser.add_argument("--compose-mode", type=str, default="tile", choices=["tile", "direct"], help="Composition mode")
    parser.add_argument("--tile-size", type=int, default=64, help="Tile source size for compose-mode=tile")
    parser.add_argument("--no-random-tiling", action="store_true", help="Disable randomization per tile block")
    parser.add_argument("--no-color-match", action="store_true", help="Disable local color matching")
    parser.add_argument("--ring-width", type=int, default=11, help="Ring width for local color transfer")
    parser.add_argument("--feather-kernel", type=int, default=15, help="Feather kernel size")
    parser.add_argument("--feather-sigma", type=float, default=3.0, help="Feather sigma")
    parser.add_argument("--input-pad", type=int, default=16, help="Reflect pad pixels before generator")
    parser.add_argument("--edge-crop", type=int, default=8, help="Crop border pixels from generated pattern then resize back")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)

    if args.checkpoint.strip():
        checkpoint_path = _resolve_checkpoint_arg(args.checkpoint)
    else:
        latest = _find_latest_checkpoint(ROOT / "checkpoints")
        if latest is None:
            raise FileNotFoundError("No checkpoint found in ./checkpoints. Please provide --checkpoint.")
        checkpoint_path = latest

    if not checkpoint_path.exists():
        available = sorted((ROOT / "checkpoints").glob("checkpoint_*.pt"))
        available_names = [p.name for p in available]
        hint = (
            f" Available in ./checkpoints: {available_names}"
            if available_names
            else " No checkpoint files found in ./checkpoints."
        )
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}.{hint}")

    service = InferenceService(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        checkpoint_dir=ROOT / "checkpoints",
        device_policy=args.device,
        compose_mode=args.compose_mode,
        tile_size=int(args.tile_size),
        random_tiling=(not bool(args.no_random_tiling)),
        color_match=(not bool(args.no_color_match)),
        ring_width=int(args.ring_width),
        feather_kernel=int(args.feather_kernel),
        feather_sigma=float(args.feather_sigma),
        input_pad=int(args.input_pad),
        edge_crop=int(args.edge_crop),
    )
    app = create_app(service)

    print(f"[webapp] checkpoint={service.checkpoint_path}")
    print(f"[webapp] checkpoint_count={len(service.available_checkpoint_paths)}")
    print(f"[webapp] environment_samples={len(service.environment_samples)} dir={service.annotations_dir}")
    print(
        f"[webapp] default_device={service.default_device} image_size={service.image_size} "
        f"compose_mode={service.compose_mode} tile_size={service.tile_size} "
        f"random_tiling={service.random_tiling} color_match={service.color_match} "
        f"feather_kernel={service.feather_kernel} feather_sigma={service.feather_sigma:.2f} "
        f"input_pad={service.input_pad} edge_crop={service.edge_crop}"
    )
    print(f"[webapp] url=http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
