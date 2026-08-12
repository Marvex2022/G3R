from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import random
import numpy as np
import torch
from PIL import Image

from .gaussian import NeuralGaussianState, load_component_npz, initialize_neural_gaussians


@dataclass
class CameraView:
    image_path: Path
    K: torch.Tensor          # [3,3]
    w2c: torch.Tensor        # [4,4]
    width: int
    height: int
    split: str
    frame_id: int
    actor_to_world: dict[str, torch.Tensor]

    def camera_center(self) -> torch.Tensor:
        c2w = torch.linalg.inv(self.w2c)
        return c2w[:3, 3]

    def load_image(self, device: str | torch.device, dtype=torch.float32) -> torch.Tensor:
        img = Image.open(self.image_path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).to(device=device, dtype=dtype)


@dataclass
class SceneTemplate:
    name: str
    root: Path
    background_npz: Path
    actor_npzs: dict[str, Path]
    sky_npz: Path | None
    sky_grid_shape: tuple[int, int] | None
    views: list[CameraView]

    def source_indices(self) -> list[int]:
        return [i for i, v in enumerate(self.views) if v.split == "source"]

    def target_indices(self) -> list[int]:
        return [i for i, v in enumerate(self.views) if v.split == "target"]

    def instantiate(self, device: str | torch.device, point_limit: int | None = None, seed: int = 0) -> "SceneState":
        bg = load_component_npz(self.background_npz, device=device, name="background")
        actors = {
            aid: load_component_npz(p, device=device, name="actor", actor_id=aid)
            for aid, p in self.actor_npzs.items()
        }
        sky = None
        if self.sky_npz is not None:
            sky = load_component_npz(
                self.sky_npz,
                device=device,
                name="sky",
                grid_shape=self.sky_grid_shape,
            )
        state = SceneState(bg, actors, sky)
        if point_limit is not None:
            state = state.subsample_static_dynamic(point_limit, seed=seed)
        return state


@dataclass
class SceneState:
    background: NeuralGaussianState
    actors: dict[str, NeuralGaussianState]
    sky: NeuralGaussianState | None = None

    def components(self) -> list[NeuralGaussianState]:
        xs = [self.background] + list(self.actors.values())
        if self.sky is not None:
            xs.append(self.sky)
        return xs

    def detach(self) -> "SceneState":
        return SceneState(
            self.background.detach(),
            {k: v.detach() for k, v in self.actors.items()},
            self.sky.detach() if self.sky is not None else None,
        )

    def requires_grad_(self, flag: bool = True) -> "SceneState":
        for x in self.components():
            x.requires_grad_(flag)
        return self

    def subsample_static_dynamic(self, limit: int, seed: int = 0) -> "SceneState":
        """Subsample the total background+actor point count; sky is kept intact."""
        pieces = [("background", None, self.background)] + [("actor", k, v) for k, v in self.actors.items()]
        total = sum(x.features.shape[0] for _, _, x in pieces)
        if total <= limit:
            return self
        g = torch.Generator(device=self.background.features.device)
        g.manual_seed(seed)
        ratio = limit / float(total)

        def sub(x: NeuralGaussianState) -> NeuralGaussianState:
            n = x.features.shape[0]
            m = max(1, int(round(n * ratio)))
            idx = torch.randperm(n, generator=g, device=x.features.device)[:m]
            return NeuralGaussianState(x.features[idx], x.name, x.actor_id, x.grid_shape)

        return SceneState(sub(self.background), {k: sub(v) for k, v in self.actors.items()}, self.sky)


def _resolve(root: Path, p: str | None) -> Path | None:
    if p is None:
        return None
    q = Path(p)
    return q if q.is_absolute() else root / q


def load_scene_manifest(path: str | Path) -> SceneTemplate:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    root = path.parent
    comps = d["components"]
    background_npz = _resolve(root, comps["background"])
    actor_npzs = {str(k): _resolve(root, v) for k, v in comps.get("actors", {}).items()}
    sky_info = comps.get("sky")
    sky_npz = None
    sky_shape = None
    if isinstance(sky_info, str):
        sky_npz = _resolve(root, sky_info)
    elif isinstance(sky_info, dict):
        sky_npz = _resolve(root, sky_info.get("path"))
        if sky_info.get("height") and sky_info.get("width"):
            sky_shape = (int(sky_info["height"]), int(sky_info["width"]))

    views: list[CameraView] = []
    for i, v in enumerate(d["views"]):
        image_path = _resolve(root, v["image"])
        K = torch.tensor(v["K"], dtype=torch.float32).reshape(3, 3)
        w2c = torch.tensor(v["w2c"], dtype=torch.float32).reshape(4, 4)
        actor_tf = {
            str(k): torch.tensor(T, dtype=torch.float32).reshape(4, 4)
            for k, T in v.get("actor_to_world", {}).items()
        }
        if "width" in v and "height" in v:
            width, height = int(v["width"]), int(v["height"])
        else:
            with Image.open(image_path) as im:
                width, height = im.size
        views.append(
            CameraView(
                image_path=image_path,
                K=K,
                w2c=w2c,
                width=width,
                height=height,
                split=v.get("split", "source"),
                frame_id=int(v.get("frame_id", i)),
                actor_to_world=actor_tf,
            )
        )
    return SceneTemplate(
        name=d.get("name", path.stem),
        root=root,
        background_npz=background_npz,
        actor_npzs=actor_npzs,
        sky_npz=sky_npz,
        sky_grid_shape=sky_shape,
        views=views,
    )


def load_manifest_list(path: str | Path) -> list[SceneTemplate]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if isinstance(d, list):
        items = d
    else:
        items = d.get("scenes", [])
    return [load_scene_manifest((path.parent / p).resolve()) for p in items]


def select_training_views(
    scene: SceneTemplate,
    n_source: int,
    n_target: int,
    mode: str = "frame",
    rng: random.Random | None = None,
) -> tuple[list[int], list[int]]:
    rng = rng or random
    src = scene.source_indices()
    tgt = scene.target_indices()
    if not tgt:
        raise ValueError(f"Scene {scene.name} has no target views")
    anchor = rng.choice(tgt)
    anchor_view = scene.views[anchor]

    if mode == "camera_distance":
        c = anchor_view.camera_center()
        score = lambda i: float(torch.linalg.norm(scene.views[i].camera_center() - c))
    else:
        score = lambda i: abs(scene.views[i].frame_id - anchor_view.frame_id)

    src_sel = sorted(src, key=score)[: min(n_source, len(src))]
    tgt_sel = sorted(tgt, key=score)[: min(n_target, len(tgt))]
    return src_sel, tgt_sel
