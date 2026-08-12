from __future__ import annotations

from dataclasses import dataclass
import torch

from .data import CameraView, SceneState
from .gaussian import GaussianDecoder, GaussianParams
from .math_utils import apply_rigid_to_gaussians


@dataclass
class RenderOutput:
    rgb: torch.Tensor    # [H,W,3]
    alpha: torch.Tensor  # [H,W,1]


class GSplatRenderer:
    """Differentiable Gaussian renderer using modern gsplat as the backend.

    G3R used a Taichi 3DGS implementation. gsplat is substituted here because it
    provides the same differentiable Gaussian primitive interface and is easier to
    maintain on modern PyTorch/CUDA stacks. This changes the rasterizer backend,
    not the G3R learned-optimizer algorithm.
    """

    def __init__(self, packed: bool = True, radius_clip: float = 0.0, near: float = 0.01, far: float = 1e5):
        self.packed = packed
        self.radius_clip = radius_clip
        self.near = near
        self.far = far

    @staticmethod
    def _actor_world_params(p: GaussianParams, T: torch.Tensor) -> GaussianParams:
        means, quats = apply_rigid_to_gaussians(p.means, p.quats, T)
        return GaussianParams(means, p.scales, quats, p.colors, p.opacities)

    def compose(self, state: SceneState, decoder: GaussianDecoder, view: CameraView) -> GaussianParams:
        pieces: list[GaussianParams] = [decoder(state.background.features)]
        for actor_id, actor_state in state.actors.items():
            p = decoder(actor_state.features)
            T = view.actor_to_world.get(actor_id)
            if T is None:
                # Actor absent/not visible in this frame: skip it instead of assuming identity.
                continue
            T = T.to(device=p.means.device, dtype=p.means.dtype)
            pieces.append(self._actor_world_params(p, T))
        if state.sky is not None:
            pieces.append(decoder(state.sky.features))
        out = pieces[0]
        return out.cat(pieces[1:]) if len(pieces) > 1 else out

    def render(self, state: SceneState, decoder: GaussianDecoder, view: CameraView) -> RenderOutput:
        try:
            from gsplat import rasterization
        except Exception as e:  # pragma: no cover
            raise ImportError("GSplatRenderer requires `pip install gsplat`.") from e

        p = self.compose(state, decoder, view)
        device = p.means.device
        viewmat = view.w2c.to(device=device, dtype=p.means.dtype)[None]
        K = view.K.to(device=device, dtype=p.means.dtype)[None]
        rgb, alpha, _ = rasterization(
            means=p.means,
            quats=p.quats,
            scales=p.scales,
            opacities=p.opacities,
            colors=p.colors,
            viewmats=viewmat,
            Ks=K,
            width=view.width,
            height=view.height,
            near_plane=self.near,
            far_plane=self.far,
            radius_clip=self.radius_clip,
            packed=self.packed,
            sh_degree=None,
            render_mode="RGB",
        )
        return RenderOutput(rgb=rgb[0], alpha=alpha[0])
