from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def logit(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x) - torch.log1p(-x)


def normalize_quaternion(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(q, p=2, dim=-1, eps=eps)


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product for wxyz quaternions, broadcast over leading dims."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ),
        dim=-1,
    )


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert (..., 3, 3) rotation matrices to normalized wxyz quaternions.

    Uses a branch-free numerically stable construction based on matrix diagonals.
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (...,3,3), got {tuple(R.shape)}")
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    qw = 0.5 * torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=0.0))
    qx = 0.5 * torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=0.0))
    qy = 0.5 * torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=0.0))
    qz = 0.5 * torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=0.0))
    qx = torch.copysign(qx, m21 - m12)
    qy = torch.copysign(qy, m02 - m20)
    qz = torch.copysign(qz, m10 - m01)
    return normalize_quaternion(torch.stack((qw, qx, qy, qz), dim=-1))


def transform_points(T: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
    """Apply a 4x4 rigid transform to Nx3 points."""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    return xyz @ R.transpose(-1, -2) + t


def apply_rigid_to_gaussians(
    means: torch.Tensor,
    quats: torch.Tensor,
    T: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    world_means = transform_points(T, means)
    rq = rotation_matrix_to_quaternion(T[:3, :3]).expand_as(quats)
    world_quats = normalize_quaternion(quaternion_multiply(rq, quats))
    return world_means, world_quats


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Standard sinusoidal positional encoding for normalized or integer timesteps."""
    if t.ndim == 0:
        t = t[None]
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half, 1))
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def cosine_ddim_gamma(step: int, total_steps: int, s: float = 0.008, floor: float = 0.0) -> float:
    """A normalized squared-cosine decay used as the G3R update scale.

    The paper says it uses a cosine scheduler "from DDIM" but does not publish the
    exact scalar formula/amplitude. This implementation follows the commonly used
    squared-cosine alpha-bar schedule and normalizes gamma(0)=1.
    """
    if total_steps <= 1:
        return 1.0
    x = min(max(step / float(total_steps - 1), 0.0), 1.0)
    f = math.cos(((x + s) / (1.0 + s)) * math.pi / 2.0) ** 2
    f0 = math.cos((s / (1.0 + s)) * math.pi / 2.0) ** 2
    return floor + (1.0 - floor) * (f / f0)
