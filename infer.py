#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import torch

from g3r.config import load_config
from g3r.data import load_scene_manifest
from g3r.engine import build_modules, reconstruct_scene
from g3r.io_utils import save_state_npz, save_rendered_views
from g3r.renderer import GSplatRenderer


def main():
    p = argparse.ArgumentParser("G3R reconstruction inference")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--turbo", action="store_true")
    p.add_argument("--render-targets", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    modules = build_modules(cfg)
    modules.load_state_dict(ckpt["modules"])
    scene = load_scene_manifest(args.scene)
    state = reconstruct_scene(cfg, scene, modules, args.device, turbo=args.turbo)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    save_state_npz(state, modules.decoder, out / "gaussians")
    if args.render_targets:
        save_rendered_views(
            state,
            scene,
            modules.decoder,
            out / "renders",
            indices=scene.target_indices(),
            renderer=GSplatRenderer(**cfg.get("renderer", {})),
        )


if __name__ == "__main__":
    main()
