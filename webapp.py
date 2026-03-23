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
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from flask import Flask, jsonify, render_template, request, send_file


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import GeneratorUNet


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


class InferenceService:
    def __init__(
        self,
        checkpoint_path: Path,
        config_path: Path,
        device_policy: str = "auto",
        compose_mode: str = "tile",
        tile_size: int = 64,
        random_tiling: bool = True,
        color_match: bool = True,
        ring_width: int = 11,
        feather_kernel: int = 15,
        feather_sigma: float = 3.0,
    ):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

        config = self._load_config(config_path)
        self.image_size = int(config.get("hcas_specific", {}).get("image_size", 256))
        self.compose_mode = str(compose_mode).strip().lower()
        self.tile_size = int(max(4, tile_size))
        self.random_tiling = bool(random_tiling)
        self.color_match = bool(color_match)
        self.ring_width = int(max(1, ring_width))
        self.feather_kernel = int(max(1, feather_kernel))
        self.feather_sigma = float(max(0.1, feather_sigma))

        self.device = _resolve_device(device_policy)
        self.generator = self._load_generator(checkpoint_path, self.device)

    @staticmethod
    def _extract_center_tile(pattern_t: torch.Tensor, tile_size: int) -> torch.Tensor:
        _, _, h, w = pattern_t.shape
        t = max(4, min(int(tile_size), int(h), int(w)))
        y0 = (int(h) - t) // 2
        x0 = (int(w) - t) // 2
        return pattern_t[:, :, y0 : y0 + t, x0 : x0 + t]

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
    def _load_generator(checkpoint_path: Path, device: torch.device) -> GeneratorUNet:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        gen_state = data.get("generator_state")
        if not isinstance(gen_state, dict):
            raise RuntimeError("Checkpoint does not contain generator_state")

        model = GeneratorUNet().to(device)
        model.load_state_dict(gen_state, strict=True)
        model.eval()
        return model

    def infer(
        self,
        image_bgr: np.ndarray,
        polygon_points: list[list[float]],
        *,
        compose_mode: str | None = None,
        tile_size: int | None = None,
    ) -> dict[str, str | int | float]:
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

        image_t = image_t.unsqueeze(0).to(self.device)
        mask_t = mask_t.unsqueeze(0).to(self.device)

        mode = (compose_mode or self.compose_mode).strip().lower()
        if mode not in {"direct", "tile"}:
            mode = self.compose_mode
        tsize = int(max(4, tile_size if tile_size is not None else self.tile_size))

        with torch.no_grad():
            pattern_direct_t = self.generator(image_t)
            tile_src_t = self._extract_center_tile(pattern_direct_t, tsize)
            if mode == "tile":
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

        env_u8 = (image_t[0].detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        pat_u8 = (tile_src_t[0].detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        pat_canvas_u8 = (
            pattern_used_t[0].detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0
        ).round().astype(np.uint8)
        comp_u8 = (composite_t[0].detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        mask_u8 = (mask_t[0, 0].detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)

        env_bgr = cv2.cvtColor(env_u8, cv2.COLOR_RGB2BGR)
        pat_bgr = cv2.cvtColor(pat_u8, cv2.COLOR_RGB2BGR)
        pat_canvas_bgr = cv2.cvtColor(pat_canvas_u8, cv2.COLOR_RGB2BGR)
        comp_bgr = cv2.cvtColor(comp_u8, cv2.COLOR_RGB2BGR)
        mask_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)

        return {
            "environment": _to_base64_png(env_bgr),
            "mask": _to_base64_png(mask_bgr),
            "pattern": _to_base64_png(pat_bgr),
            "pattern_canvas": _to_base64_png(pat_canvas_bgr),
            "composite": _to_base64_png(comp_bgr),
            "width": int(self.image_size),
            "height": int(self.image_size),
            "mask_area_ratio": float(mask.mean()),
            "compose_mode": mode,
            "tile_size": int(tsize),
            "random_tiling": bool(self.random_tiling),
            "color_match": bool(self.color_match),
            "feather_kernel": int(self.feather_kernel),
            "feather_sigma": float(self.feather_sigma),
        }


def _find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    candidates = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
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


def create_app(service: InferenceService) -> Flask:
    app = Flask(__name__, template_folder=str(ROOT / "web" / "templates"), static_folder=str(ROOT / "web" / "static"))

    model_glb_path = ROOT / "long_sleeve_t-_shirt.glb"

    @app.get("/")
    def index() -> str:
        return render_template(
            "index.html",
            checkpoint_name=service.checkpoint_path.name,
            image_size=service.image_size,
            device=str(service.device),
            compose_mode=service.compose_mode,
            tile_size=service.tile_size,
            random_tiling=service.random_tiling,
            color_match=service.color_match,
            feather_kernel=service.feather_kernel,
            feather_sigma=service.feather_sigma,
            model_glb_url="/model/long_sleeve_t-_shirt.glb",
        )

    @app.get("/model/long_sleeve_t-_shirt.glb")
    def model_glb() -> Any:
        if not model_glb_path.exists():
            return jsonify({"error": f"Model file not found: {model_glb_path.name}"}), 404
        return send_file(model_glb_path, mimetype="model/gltf-binary")

    @app.post("/api/infer")
    def infer() -> tuple[Any, int] | Any:
        file = request.files.get("image")
        points_raw = request.form.get("points", "[]")

        if file is None:
            return jsonify({"error": "Missing image file."}), 400

        try:
            points = json.loads(points_raw)
        except Exception:
            return jsonify({"error": "Invalid points JSON."}), 400

        if not isinstance(points, list):
            return jsonify({"error": "Points must be a list."}), 400

        data = np.frombuffer(file.read(), dtype=np.uint8)
        image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image_bgr is None:
            return jsonify({"error": "Failed to decode uploaded image."}), 400

        try:
            compose_mode = str(request.form.get("compose_mode", service.compose_mode)).strip().lower()
            tile_size_raw = request.form.get("tile_size", str(service.tile_size))
            try:
                tile_size = int(tile_size_raw)
            except Exception:
                tile_size = int(service.tile_size)

            result = service.infer(image_bgr, points, compose_mode=compose_mode, tile_size=tile_size)
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
        device_policy=args.device,
        compose_mode=args.compose_mode,
        tile_size=int(args.tile_size),
        random_tiling=(not bool(args.no_random_tiling)),
        color_match=(not bool(args.no_color_match)),
        ring_width=int(args.ring_width),
        feather_kernel=int(args.feather_kernel),
        feather_sigma=float(args.feather_sigma),
    )
    app = create_app(service)

    print(f"[webapp] checkpoint={service.checkpoint_path}")
    print(
        f"[webapp] device={service.device} image_size={service.image_size} "
        f"compose_mode={service.compose_mode} tile_size={service.tile_size} "
        f"random_tiling={service.random_tiling} color_match={service.color_match} "
        f"feather_kernel={service.feather_kernel} feather_sigma={service.feather_sigma:.2f}"
    )
    print(f"[webapp] url=http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
