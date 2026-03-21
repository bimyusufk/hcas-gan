"""DeepGaze saliency wrapper with real backend + safe fallback.

Design goals:
- Prefer real DeepGazeIII backend via ``torch.hub`` when available.
- Keep DeepGaze parameters frozen while preserving gradient flow to input image.
- Provide robust fallback proxy backend so trainer remains runnable in constrained setups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

BackendName = Literal["auto", "deepgazeiii_torchhub", "proxy"]
CenterBiasMode = Literal["zeros", "gaussian"]
OutputMode = Literal["log_density", "probability", "density", "sigmoid"]
InputRangeMode = Literal["0_1", "0_255"]


@dataclass
class DeepGazeBackendInfo:
    requested_backend: str
    active_backend: str
    model_name: str
    using_fallback: bool
    fallback_reason: str | None


class DeepGazeWrapper(nn.Module):
    """Saliency wrapper for HCAS-GAN.

    Args:
        backend: Backend strategy: ``auto`` / ``deepgazeiii_torchhub`` / ``proxy``.
        model_name: Model name from DeepGaze torch hub repo (e.g. ``DeepGazeIII``).
        pretrained: Load pretrained weights for DeepGaze model.
        trust_repo: Passed to ``torch.hub.load``.
        repo: Hub repository ID.
        centerbias_mode: ``zeros`` (uniform prior) or ``gaussian``.
        fixation_history_length: Number of synthetic fixation history points for DeepGazeIII.
        output_mode: Output transform from model log-density.
        input_range: Input scaling mode expected by backend.
        allow_fallback: If True, fallback to proxy when real backend load fails.
        verbose: Print backend/fallback status on initialization.
    """

    def __init__(
        self,
        *,
        backend: BackendName | str = "auto",
        model_name: str = "DeepGazeIII",
        pretrained: bool = True,
        trust_repo: bool = True,
        repo: str = "matthias-k/DeepGaze",
        centerbias_mode: CenterBiasMode | str = "zeros",
        fixation_history_length: int = 4,
        output_mode: OutputMode | str = "density",
        input_range: InputRangeMode | str = "0_1",
        allow_fallback: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        self.requested_backend = str(backend)
        self.model_name = str(model_name)
        self.pretrained = bool(pretrained)
        self.trust_repo = bool(trust_repo)
        self.repo = str(repo)
        self.centerbias_mode = str(centerbias_mode)
        self.fixation_history_length = int(fixation_history_length)
        self.output_mode = str(output_mode)
        self.input_range = str(input_range)
        self.allow_fallback = bool(allow_fallback)
        self.verbose = bool(verbose)

        if self.fixation_history_length <= 0:
            raise ValueError("fixation_history_length must be > 0")
        if self.centerbias_mode not in ("zeros", "gaussian"):
            raise ValueError(f"Unsupported centerbias_mode: {self.centerbias_mode}")
        if self.output_mode not in ("log_density", "probability", "density", "sigmoid"):
            raise ValueError(f"Unsupported output_mode: {self.output_mode}")
        if self.input_range not in ("0_1", "0_255"):
            raise ValueError(f"Unsupported input_range: {self.input_range}")

        self._real_backend: nn.Module | None = None
        self._proxy_backend: nn.Module | None = None

        self.active_backend: str = ""
        self.fallback_reason: str | None = None
        self.using_fallback: bool = False

        self._initialize_backend()

    def _initialize_backend(self) -> None:
        backend = self.requested_backend.lower()

        if backend not in ("auto", "deepgazeiii_torchhub", "proxy"):
            raise ValueError(f"Unsupported backend: {self.requested_backend}")

        if backend in ("auto", "deepgazeiii_torchhub"):
            try:
                self._real_backend = self._load_deepgaze_torchhub()
                self.active_backend = "deepgazeiii_torchhub"
                if self.verbose:
                    print(f"[DeepGazeWrapper] Using backend: {self.active_backend} ({self.model_name})")
                return
            except Exception as exc:
                if not self.allow_fallback and backend == "deepgazeiii_torchhub":
                    raise
                self.fallback_reason = f"{type(exc).__name__}: {exc}"

        self._proxy_backend = self._build_proxy_backend()
        self.active_backend = "proxy"
        self.using_fallback = backend != "proxy"
        if self.verbose:
            if self.fallback_reason:
                print(
                    "[DeepGazeWrapper] Falling back to proxy backend. "
                    f"Reason: {self.fallback_reason}"
                )
            else:
                print("[DeepGazeWrapper] Using backend: proxy")

    def _load_deepgaze_torchhub(self) -> nn.Module:
        model = torch.hub.load(
            self.repo,
            self.model_name,
            pretrained=self.pretrained,
            trust_repo=self.trust_repo,
        )
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    @staticmethod
    def _build_proxy_backend() -> nn.Module:
        proxy = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        for p in proxy.parameters():
            p.requires_grad = False
        return proxy

    def get_backend_info(self) -> DeepGazeBackendInfo:
        return DeepGazeBackendInfo(
            requested_backend=self.requested_backend,
            active_backend=self.active_backend,
            model_name=self.model_name,
            using_fallback=self.using_fallback,
            fallback_reason=self.fallback_reason,
        )

    def _prepare_input(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"DeepGaze input must be [N,3,H,W], got: {tuple(image.shape)}")
        x = image.float()
        if self.input_range == "0_1":
            x = x * 255.0
        return x

    def _build_centerbias(self, image: torch.Tensor) -> torch.Tensor:
        n, _, h, w = image.shape
        if self.centerbias_mode == "zeros":
            return torch.zeros((n, h, w), device=image.device, dtype=image.dtype)

        ys = torch.linspace(-1.0, 1.0, steps=h, device=image.device, dtype=image.dtype)
        xs = torch.linspace(-1.0, 1.0, steps=w, device=image.device, dtype=image.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        sigma = 0.35
        log_gauss = -0.5 * ((xx / sigma) ** 2 + (yy / sigma) ** 2)
        log_gauss = log_gauss - torch.logsumexp(log_gauss.reshape(-1), dim=0)
        return log_gauss.unsqueeze(0).repeat(n, 1, 1)

    def _build_fixation_history(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n, _, h, w = image.shape
        x_center = (float(w) - 1.0) * 0.5
        y_center = (float(h) - 1.0) * 0.5
        x_hist = torch.full(
            (n, self.fixation_history_length),
            fill_value=x_center,
            device=image.device,
            dtype=image.dtype,
        )
        y_hist = torch.full(
            (n, self.fixation_history_length),
            fill_value=y_center,
            device=image.device,
            dtype=image.dtype,
        )
        return x_hist, y_hist

    def _format_output(self, raw_output: torch.Tensor) -> torch.Tensor:
        if raw_output.ndim == 3:
            raw_output = raw_output.unsqueeze(1)
        if raw_output.ndim != 4:
            raise ValueError(f"Unexpected DeepGaze output shape: {tuple(raw_output.shape)}")

        if self.output_mode == "log_density":
            return raw_output
        if self.output_mode == "sigmoid":
            return torch.sigmoid(raw_output)

        prob = torch.exp(raw_output)
        if self.output_mode == "probability":
            return prob

        # density mode: convert probability mass map to per-pixel density scale.
        h, w = raw_output.shape[-2:]
        return prob * float(h * w)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.active_backend == "proxy":
            if self._proxy_backend is None:
                raise RuntimeError("Proxy backend not initialized.")
            return self._proxy_backend(image)

        if self._real_backend is None:
            raise RuntimeError("Real DeepGaze backend not initialized.")

        x = self._prepare_input(image)
        centerbias = self._build_centerbias(x)
        x_hist, y_hist = self._build_fixation_history(x)

        # Keep graph for input gradients; only DeepGaze parameters are frozen.
        try:
            raw = self._real_backend(x, centerbias, x_hist, y_hist)
        except TypeError:
            # Some hub variants do not require scanpath history.
            raw = self._real_backend(x, centerbias)

        return self._format_output(raw)

    def train(self, mode: bool = True):
        super().train(mode)
        if self._real_backend is not None:
            self._real_backend.eval()
        return self
