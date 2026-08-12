from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .math_utils import logit, normalize_quaternion

GAUSSIAN_DIM = 14
LATENT_DIM = 32
STATE_DIM = GAUSSIAN_DIM + LATENT_DIM  # paper: C=46


@dataclass
class GaussianParams:
    means: torch.Tensor      # [N,3]
    scales: torch.Tensor     # [N,3], positive
    quats: torch.Tensor      # [N,4], wxyz
    colors: torch.Tensor     # [N,3], [0,1]
    opacities: torch.Tensor  # [N], [0,1]

    def cat(self, others: list["GaussianParams"]) -> "GaussianParams":
        xs = [self] + others
        return GaussianParams(
            means=torch.cat([x.means for x in xs], 0),
            scales=torch.cat([x.scales for x in xs], 0),
            quats=torch.cat([x.quats for x in xs], 0),
            colors=torch.cat([x.colors for x in xs], 0),
            opacities=torch.cat([x.opacities for x in xs], 0),
        )


@dataclass
class NeuralGaussianState:
    """A scene component represented by C=46 neural-Gaussian features."""
    features: torch.Tensor  # [N,46]
    name: str = "background"
    actor_id: Optional[str] = None
    grid_shape: Optional[tuple[int, int]] = None  # for sky equirectangular grid

    def detach(self) -> "NeuralGaussianState":
        return NeuralGaussianState(
            self.features.detach(), self.name, self.actor_id, self.grid_shape
        )

    def requires_grad_(self, flag: bool = True) -> "NeuralGaussianState":
        self.features.requires_grad_(flag)
        return self

    def to(self, device: torch.device | str) -> "NeuralGaussianState":
        return NeuralGaussianState(
            self.features.to(device), self.name, self.actor_id, self.grid_shape
        )


class GaussianDecoder(nn.Module):
    """Paper f_mlp: one linear layer + tanh + skip on the first 14 channels.

    We store the 14 Gaussian attributes in an unconstrained parameterization:
      xyz, log(scale), raw quaternion(wxyz), color logits, opacity logit.
    This is an engineering choice because the paper does not specify the exact
    positivity/range parameterization used inside their Taichi 3DGS backend.
    """

    def __init__(self, state_dim: int = STATE_DIM):
        super().__init__()
        self.proj = nn.Linear(state_dim, GAUSSIAN_DIM)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def raw(self, h: torch.Tensor) -> torch.Tensor:
        residual = torch.tanh(self.proj(h))
        return h[..., :GAUSSIAN_DIM] + residual

    def forward(self, h: torch.Tensor) -> GaussianParams:
        raw = self.raw(h)
        means = raw[..., 0:3]
        scales = torch.exp(raw[..., 3:6]).clamp(1e-5, 1e3)
        quats = normalize_quaternion(raw[..., 6:10])
        colors = torch.sigmoid(raw[..., 10:13])
        opacities = torch.sigmoid(raw[..., 13])
        return GaussianParams(means, scales, quats, colors, opacities)


def _third_neighbor_scale(xyz: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree
    if len(xyz) < 4:
        return np.full((len(xyz),), 0.05, dtype=np.float32)
    tree = cKDTree(xyz)
    d, _ = tree.query(xyz, k=4, workers=-1)
    # k=4 includes self at index 0; index 3 is the third non-self neighbor.
    return np.maximum(d[:, 3], 1e-4).astype(np.float32)


def initialize_neural_gaussians(
    xyz: np.ndarray | torch.Tensor,
    rgb: np.ndarray | torch.Tensor | None = None,
    scale: np.ndarray | torch.Tensor | None = None,
    latent_std: float = 0.01,
    opacity: float = 0.7,
    device: str | torch.device = "cpu",
    name: str = "background",
    actor_id: str | None = None,
    grid_shape: tuple[int, int] | None = None,
) -> NeuralGaussianState:
    xyz_np = xyz.detach().cpu().numpy() if torch.is_tensor(xyz) else np.asarray(xyz)
    xyz_np = xyz_np.astype(np.float32)
    n = len(xyz_np)
    if scale is None:
        scale_np = _third_neighbor_scale(xyz_np)
    else:
        scale_np = scale.detach().cpu().numpy() if torch.is_tensor(scale) else np.asarray(scale)
        if scale_np.ndim == 2:
            scale_np = np.mean(scale_np, axis=-1)
        scale_np = scale_np.astype(np.float32)
    if rgb is None:
        # Paper-faithful default: positions/scales/rotation/opacity are prescribed,
        # while the remaining Gaussian attributes are randomly initialized.
        rgb_t = torch.rand((n, 3), dtype=torch.float32, device=device).clamp(1e-4, 1 - 1e-4)
    else:
        # Optional engineering shortcut for datasets that already provide point colors.
        rgb_np = rgb.detach().cpu().numpy() if torch.is_tensor(rgb) else np.asarray(rgb)
        rgb_np = rgb_np.astype(np.float32)
        if rgb_np.max(initial=1.0) > 1.5:
            rgb_np /= 255.0
        rgb_np = np.clip(rgb_np, 1e-4, 1 - 1e-4)
        rgb_t = torch.from_numpy(rgb_np).to(device)

    h = torch.zeros((n, STATE_DIM), dtype=torch.float32, device=device)
    h[:, 0:3] = torch.from_numpy(xyz_np).to(device)
    s = torch.from_numpy(scale_np).to(device).clamp_min(1e-5)
    h[:, 3:6] = torch.log(s[:, None].expand(-1, 3))
    h[:, 6] = 1.0  # identity quaternion wxyz
    h[:, 10:13] = logit(rgb_t)
    h[:, 13] = math.log(opacity / (1.0 - opacity))
    h[:, GAUSSIAN_DIM:] = latent_std * torch.randn((n, LATENT_DIM), device=device)
    return NeuralGaussianState(h, name=name, actor_id=actor_id, grid_shape=grid_shape)


def load_component_npz(path: str | Path, device: str | torch.device = "cpu", **kwargs) -> NeuralGaussianState:
    data = np.load(path)
    xyz = data["xyz"]
    rgb = data["rgb"] if "rgb" in data else None
    scale = data["scale"] if "scale" in data else None
    return initialize_neural_gaussians(xyz, rgb, scale, device=device, **kwargs)


def make_sky_sphere(
    center: torch.Tensor,
    height: int,
    width: int,
    radius: float = 2048.0,
    lat_north_deg: float = 30.0,
    lat_south_deg: float = -15.0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Generate cropped equirectangular sky points as described in the supplement."""
    lat = torch.linspace(math.radians(lat_south_deg), math.radians(lat_north_deg), height, device=device)
    lon = torch.linspace(-math.pi, math.pi, width, device=device)
    latg, long = torch.meshgrid(lat, lon, indexing="ij")
    x = torch.cos(latg) * torch.cos(long)
    y = torch.cos(latg) * torch.sin(long)
    z = torch.sin(latg)
    xyz = torch.stack([x, y, z], dim=-1).reshape(-1, 3) * radius
    return xyz + center.reshape(1, 3)
