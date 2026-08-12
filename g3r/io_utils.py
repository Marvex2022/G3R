from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image

from .data import SceneState, SceneTemplate
from .gaussian import GaussianDecoder, GaussianParams
from .renderer import GSplatRenderer


def save_state_npz(state: SceneState, decoder: GaussianDecoder, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_one(name: str, features: torch.Tensor):
        with torch.no_grad():
            p = decoder(features)
        np.savez_compressed(
            out_dir / f"{name}.npz",
            neural_features=features.detach().cpu().numpy(),
            xyz=p.means.detach().cpu().numpy(),
            scale=p.scales.detach().cpu().numpy(),
            quat_wxyz=p.quats.detach().cpu().numpy(),
            rgb=p.colors.detach().cpu().numpy(),
            opacity=p.opacities.detach().cpu().numpy(),
        )

    save_one("background", state.background.features)
    for aid, actor in state.actors.items():
        save_one(f"actor_{aid}", actor.features)
    if state.sky is not None:
        save_one("sky", state.sky.features)


def save_rendered_views(
    state: SceneState,
    scene: SceneTemplate,
    decoder: GaussianDecoder,
    out_dir: str | Path,
    indices: list[int] | None = None,
    renderer: GSplatRenderer | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = renderer or GSplatRenderer()
    if indices is None:
        indices = scene.target_indices() or scene.source_indices()
    for idx in indices:
        view = scene.views[idx]
        with torch.no_grad():
            rgb = renderer.render(state, decoder, view).rgb.clamp(0, 1)
        arr = (rgb.detach().cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(arr).save(out_dir / f"{idx:06d}.png")
