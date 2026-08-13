#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
from g3r.data import load_scene_manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scene")
    args = p.parse_args()
    s = load_scene_manifest(args.scene)
    print(f"name: {s.name}")
    print(f"views: {len(s.views)} source={len(s.source_indices())} target={len(s.target_indices())}")
    for name, path in [("background", s.background_npz), *[(f"actor:{k}", v) for k, v in s.actor_npzs.items()]]:
        d = np.load(path)
        print(f"{name}: {len(d['xyz']):,} points; keys={list(d.keys())}")
    if s.sky_npz:
        d = np.load(s.sky_npz)
        print(f"sky: {len(d['xyz']):,} points grid={s.sky_grid_shape}; keys={list(d.keys())}")
    for i, v in enumerate(s.views[:3]):
        print(f"view[{i}] frame={v.frame_id} split={v.split} size={v.width}x{v.height} actors={list(v.actor_to_world)}")


if __name__ == "__main__":
    main()
