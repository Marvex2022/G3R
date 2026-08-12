#!/usr/bin/env python3
"""Create the cropped spherical distant/sky scaffold described by G3R."""
from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import torch

from g3r.gaussian import make_sky_sphere


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--center", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=2048)
    p.add_argument("--radius", type=float, default=2048.0)
    p.add_argument("--lat-north", type=float, default=30.0)
    p.add_argument("--lat-south", type=float, default=-15.0)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    center = torch.tensor(args.center, dtype=torch.float32)
    xyz = make_sky_sphere(
        center,
        height=args.height,
        width=args.width,
        radius=args.radius,
        lat_north_deg=args.lat_north,
        lat_south_deg=args.lat_south,
        device="cpu",
    ).numpy()
    # For a regular spherical image grid, use local neighborhood spacing rather
    # than an expensive all-pairs/3-NN query over ~1M-4M points.
    lat_span = np.deg2rad(args.lat_north - args.lat_south)
    d_lat = args.radius * lat_span / max(args.height - 1, 1)
    d_lon = args.radius * (2.0 * np.pi) / max(args.width - 1, 1)
    scale = np.full((len(xyz),), max(min(d_lat, d_lon), 1e-4), dtype=np.float32)
    np.savez_compressed(args.output, xyz=xyz.astype(np.float32), scale=scale)
    print(f"saved sky {args.height}x{args.width} ({len(xyz):,} Gaussians) -> {args.output}")


if __name__ == "__main__":
    main()
