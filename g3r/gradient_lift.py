from __future__ import annotations

from dataclasses import dataclass
import torch

from .data import SceneState, SceneTemplate
from .gaussian import GaussianDecoder
from .renderer import GSplatRenderer


@dataclass
class SceneGradients:
    background: torch.Tensor
    actors: dict[str, torch.Tensor]
    sky: torch.Tensor | None


def _normalize_channelwise(g: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Supplement A.3: divide each raw gradient channel by its maximal absolute value.
    denom = g.detach().abs().amax(dim=0, keepdim=True).clamp_min(eps)
    return g / denom


def lift_source_images_to_gradients(
    state: SceneState,
    scene: SceneTemplate,
    source_indices: list[int],
    decoder: GaussianDecoder,
    renderer: GSplatRenderer,
    device: str | torch.device,
    normalize: bool = True,
) -> SceneGradients:
    """Render source images and accumulate d ||I-Ihat||_2^2 / dS over all views.

    We intentionally use create_graph=False and return detached gradients. G3R's
    paper does not state whether second-order differentiation is used. A full
    Hessian path through a Gaussian rasterizer is generally unsupported and would
    be prohibitively expensive for 0.8M-3M points; this stop-gradient treatment is
    the practical learned-optimizer interpretation.
    """
    comps = state.components()
    tensors = [x.features for x in comps]
    for x in tensors:
        if not x.requires_grad:
            raise ValueError("Gradient lifting requires leaf state tensors with requires_grad=True")
    accum = [torch.zeros_like(x) for x in tensors]

    for vidx in source_indices:
        view = scene.views[vidx]
        target = view.load_image(device=device, dtype=tensors[0].dtype)
        out = renderer.render(state, decoder, view)
        # Main Eq. 3 writes an L2 image norm. We use mean squared photometric loss;
        # channelwise max normalization removes its global scale before G3R-Net.
        loss = torch.mean((out.rgb - target) ** 2)
        grads = torch.autograd.grad(
            loss,
            tensors,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        for i, g in enumerate(grads):
            if g is not None:
                accum[i] += g.detach()

    if normalize:
        accum = [_normalize_channelwise(g) for g in accum]

    k = 0
    bg = accum[k]; k += 1
    actors = {}
    for aid in state.actors:
        actors[aid] = accum[k]; k += 1
    sky = accum[k] if state.sky is not None else None
    return SceneGradients(bg, actors, sky)
