from __future__ import annotations

from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F

from .gaussian import STATE_DIM
from .math_utils import sinusoidal_timestep_embedding


class PointMLPOptimizer(nn.Module):
    """Debug fallback. It does NOT reproduce G3R's sparse spatial aggregation."""

    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 256, time_dim: int = 32):
        super().__init__()
        self.time_dim = time_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2 + time_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, state_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        state: torch.Tensor,
        grad: torch.Tensor,
        step: int,
        total_steps: int,
        positions=None,
        batch_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t = torch.tensor([step / max(total_steps - 1, 1)], device=state.device)
        te = sinusoidal_timestep_embedding(t, self.time_dim).to(state.dtype).expand(state.shape[0], -1)
        return self.net(torch.cat([state, grad, te], dim=-1))


class SparseG3RNet(nn.Module):
    """Paper-faithful TorchSparse SparseResUNet42-style G3R network.

    Supplement A.3 says:
      - use SparseResUNet from TorchSparse without architecture tuning;
      - concatenate S and grad(S) as input;
      - concatenate timestep positional encoding at the last encoder layer;
      - tanh at the output.

    TorchSparse's published SparseResUNet42 uses stem=32,
    enc=[32,64,128,256], dec=[256,128,96,96]. We reproduce that topology.
    """

    def __init__(self, state_dim: int = STATE_DIM, time_dim: int = 32, voxel_size: float = 0.20):
        super().__init__()
        try:
            from torchsparse import nn as spnn
            from torchsparse.backbones.modules import (
                SparseConvBlock,
                SparseConvTransposeBlock,
                SparseResBlock,
            )
        except Exception as e:  # pragma: no cover - CUDA env dependent
            raise ImportError(
                "SparseG3RNet requires TorchSparse. Install the CUDA/PyTorch-matched "
                "wheel from https://github.com/mit-han-lab/torchsparse"
            ) from e

        self.state_dim = state_dim
        self.time_dim = time_dim
        self.voxel_size = float(voxel_size)
        in_channels = state_dim * 2

        self.stem = nn.Sequential(
            spnn.Conv3d(in_channels, 32, 3), spnn.BatchNorm(32), spnn.ReLU(True),
            spnn.Conv3d(32, 32, 3), spnn.BatchNorm(32), spnn.ReLU(True),
        )
        self.enc0 = nn.Sequential(SparseConvBlock(32, 32, 2, stride=2), SparseResBlock(32, 32, 3), SparseResBlock(32, 32, 3))
        self.enc1 = nn.Sequential(SparseConvBlock(32, 32, 2, stride=2), SparseResBlock(32, 64, 3), SparseResBlock(64, 64, 3))
        self.enc2 = nn.Sequential(SparseConvBlock(64, 64, 2, stride=2), SparseResBlock(64, 128, 3), SparseResBlock(128, 128, 3))
        self.enc3 = nn.Sequential(SparseConvBlock(128, 128, 2, stride=2), SparseResBlock(128, 256, 3), SparseResBlock(256, 256, 3))

        # First decoder sees bottleneck + timestep positional encoding.
        self.up0 = SparseConvTransposeBlock(256 + time_dim, 256, 2, stride=2)
        self.fuse0 = nn.Sequential(SparseResBlock(256 + 128, 256, 3), SparseResBlock(256, 256, 3))
        self.up1 = SparseConvTransposeBlock(256, 128, 2, stride=2)
        self.fuse1 = nn.Sequential(SparseResBlock(128 + 64, 128, 3), SparseResBlock(128, 128, 3))
        self.up2 = SparseConvTransposeBlock(128, 96, 2, stride=2)
        self.fuse2 = nn.Sequential(SparseResBlock(96 + 32, 96, 3), SparseResBlock(96, 96, 3))
        self.up3 = SparseConvTransposeBlock(96, 96, 2, stride=2)
        self.fuse3 = nn.Sequential(SparseResBlock(96 + 32, 96, 3), SparseResBlock(96, 96, 3))
        self.head = spnn.Conv3d(96, state_dim, 1)

    @staticmethod
    def _clone_with_feats(x, feats):
        from torchsparse import SparseTensor
        y = SparseTensor(feats=feats, coords=x.coords, stride=x.stride, spatial_range=x.spatial_range)
        # Preserve kernel maps needed by inverse/transposed sparse convolutions.
        if hasattr(x, "_caches"):
            y._caches = x._caches
        return y

    @staticmethod
    def _cat(a, b):
        import torchsparse
        return torchsparse.cat([a, b])

    def _voxelize(
        self,
        positions: torch.Tensor,
        feats: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None,
    ):
        """Pool duplicate points into sparse voxels and return inverse map.

        Dynamic actors are separate sparse batches. A zero-feature guard voxel,
        isolated from the actor and aligned to all four stride-2 levels, prevents
        TorchSparse from producing an empty coordinate set for tiny actors.
        """
        from torchsparse import SparseTensor
        if batch_indices is None:
            batch_indices = torch.zeros(positions.shape[0], device=positions.device, dtype=torch.int32)
            dynamic_batch = False
        else:
            batch_indices = batch_indices.to(device=positions.device, dtype=torch.int32)
            if batch_indices.shape != (positions.shape[0],):
                raise ValueError(f"batch_indices must have shape ({positions.shape[0]},)")
            dynamic_batch = True

        xyz = torch.empty_like(positions, dtype=torch.int32)
        batches = torch.unique(batch_indices, sorted=True)
        for batch_id in batches:
            mask = batch_indices == batch_id
            batch_positions = positions[mask]
            xyz[mask] = torch.floor(
                (batch_positions - batch_positions.amin(dim=0, keepdim=True)) / self.voxel_size
            ).to(torch.int32)
        coords = torch.cat([batch_indices[:, None], xyz], dim=1)
        input_count = len(coords)

        if dynamic_batch:
            guards = []
            guard_feats = []
            alignment = 16  # four stride-2 encoder levels
            for batch_id in batches:
                batch_xyz = xyz[batch_indices == batch_id]
                # For kernel=2/stride=2/padding=0, a boundary coordinate with
                # low bits 1111 survives four successive downsamplings. An
                # even maximum would be cropped by TorchSparse's output bound.
                guard_value = (int(batch_xyz.max().item()) // alignment + 1) * alignment - 1
                guards.append(
                    torch.tensor(
                        [[int(batch_id.item()), guard_value, guard_value, guard_value]],
                        device=coords.device,
                        dtype=torch.int32,
                    )
                )
                guard_feats.append(torch.zeros((1, feats.shape[1]), device=feats.device, dtype=feats.dtype))
            coords = torch.cat([coords, *guards], dim=0)
            feats = torch.cat([feats, *guard_feats], dim=0)

        uniq, inv_all = torch.unique(coords, dim=0, return_inverse=True)
        # torch.unique may reorder coordinates even when every point occupies a
        # different voxel, so always pool in unique-coordinate order and always
        # keep the inverse map back to the original point order.
        pooled = torch.zeros((uniq.shape[0], feats.shape[1]), device=feats.device, dtype=feats.dtype)
        pooled.index_add_(0, inv_all, feats)
        counts = torch.bincount(inv_all, minlength=uniq.shape[0]).to(feats.dtype).clamp_min(1)
        pooled = pooled / counts[:, None]
        return SparseTensor(feats=pooled, coords=uniq), inv_all[:input_count]

    def forward(
        self,
        state: torch.Tensor,
        grad: torch.Tensor,
        step: int,
        total_steps: int,
        positions: Optional[torch.Tensor] = None,
        batch_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if positions is None:
            positions = state[:, :3]
        x, inverse = self._voxelize(
            positions.detach(), torch.cat([state, grad], dim=-1), batch_indices=batch_indices
        )
        x0 = self.stem(x)
        x1 = self.enc0(x0)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)

        t = torch.tensor([step / max(total_steps - 1, 1)], device=state.device)
        te = sinusoidal_timestep_embedding(t, self.time_dim).to(x4.F.dtype)
        te = te.expand(x4.F.shape[0], -1)
        x4t = self._clone_with_feats(x4, torch.cat([x4.F, te], dim=-1))

        y3 = self.fuse0(self._cat(self.up0(x4t), x3))
        y2 = self.fuse1(self._cat(self.up1(y3), x2))
        y1 = self.fuse2(self._cat(self.up2(y2), x1))
        y0 = self.fuse3(self._cat(self.up3(y1), x0))
        out = torch.tanh(self.head(y0).F)
        if inverse is not None:
            out = out[inverse]
        return out


class Residual2DBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.net(x), inplace=True)


class SkyG3RNet(nn.Module):
    """2D two-residual-block optimizer for the equirectangular distant/sky grid.

    Supplement A.3 contains an apparent wording inconsistency: immediately after
    saying static background and dynamics use SparseResUNet, it says "For the
    background reconstruction network, we use a 2D CNN...". Given the preceding
    decomposition and the sphere-image sky representation, this implementation
    interprets that 2D network as the sky/distant-region optimizer.
    """

    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 96, time_dim: int = 32):
        super().__init__()
        self.time_dim = time_dim
        self.in_proj = nn.Conv2d(state_dim * 2 + time_dim, hidden, 3, padding=1)
        self.res1 = Residual2DBlock(hidden)
        self.res2 = Residual2DBlock(hidden)
        self.out = nn.Conv2d(hidden, state_dim, 1)

    def forward(self, state: torch.Tensor, grad: torch.Tensor, step: int, total_steps: int, grid_shape: tuple[int, int]) -> torch.Tensor:
        H, W = grid_shape
        if state.shape[0] != H * W:
            raise ValueError(f"Sky state has {state.shape[0]} points, grid={H}x{W}")
        x = torch.cat([state, grad], dim=-1).reshape(H, W, -1).permute(2, 0, 1)[None]
        t = torch.tensor([step / max(total_steps - 1, 1)], device=state.device)
        te = sinusoidal_timestep_embedding(t, self.time_dim).to(x.dtype)
        te = te[:, :, None, None].expand(-1, -1, H, W)
        x = torch.cat([x, te], dim=1)
        y = F.relu(self.in_proj(x), inplace=True)
        y = self.res1(y)
        y = self.res2(y)
        y = torch.tanh(self.out(y))
        return y[0].permute(1, 2, 0).reshape(-1, self.out.out_channels)


def build_optimizer_network(cfg: dict, state_dim: int = STATE_DIM) -> nn.Module:
    backend = cfg.get("backend", "torchsparse")
    if backend == "torchsparse":
        return SparseG3RNet(
            state_dim=state_dim,
            time_dim=int(cfg.get("time_dim", 32)),
            voxel_size=float(cfg.get("voxel_size", 0.20)),
        )
    if backend == "point_mlp":
        return PointMLPOptimizer(
            state_dim=state_dim,
            hidden=int(cfg.get("hidden", 256)),
            time_dim=int(cfg.get("time_dim", 32)),
        )
    raise ValueError(f"Unknown optimizer backend: {backend}")
