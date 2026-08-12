#!/usr/bin/env python3
"""Prepare a component NPZ from xyz/rgb arrays and precompute 3-NN scale.

Input NPZ must contain `xyz` [N,3], optionally `rgb` [N,3]. Output adds `scale`
[N]. This is useful because 3-NN search over millions of points is best done once.
"""
from __future__ import annotations

import argparse
import numpy as np
from scipy.spatial import cKDTree


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-points", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    d = np.load(args.input)
    xyz = np.asarray(d["xyz"], dtype=np.float32)
    rgb = np.asarray(d["rgb"], dtype=np.float32) if "rgb" in d else None
    if args.max_points and len(xyz) > args.max_points:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(xyz), args.max_points, replace=False)
        xyz = xyz[idx]
        if rgb is not None:
            rgb = rgb[idx]
    tree = cKDTree(xyz)
    dist, _ = tree.query(xyz, k=min(4, len(xyz)), workers=-1)
    if dist.ndim == 1 or dist.shape[1] < 4:
        scale = np.full(len(xyz), 0.05, dtype=np.float32)
    else:
        scale = np.maximum(dist[:, 3], 1e-4).astype(np.float32)
    kw = {"xyz": xyz, "scale": scale}
    if rgb is not None:
        kw["rgb"] = rgb
    np.savez_compressed(args.output, **kw)
    print(f"saved {len(xyz):,} points -> {args.output}")


if __name__ == "__main__":
    main()
