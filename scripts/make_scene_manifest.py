#!/usr/bin/env python3
"""Create a static-scene manifest from a simple cameras JSON.

The cameras input is a JSON list with entries:
  {"image":"...", "K":[9 numbers or 3x3], "w2c":[16 numbers or 4x4], "frame_id":0}
Even frame_id values become source views and odd values target views by default,
matching G3R's every-other-frame evaluation split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--points", required=True, help="background component NPZ")
    p.add_argument("--cameras", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    with open(args.cameras, "r", encoding="utf-8") as f:
        cams = json.load(f)
    views = []
    for i, c in enumerate(cams):
        c = dict(c)
        frame = int(c.get("frame_id", i))
        c["frame_id"] = frame
        c["split"] = c.get("split", "source" if frame % 2 == 0 else "target")
        views.append(c)
    out = {"name": args.name, "components": {"background": args.points, "actors": {}}, "views": views}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(args.output)


if __name__ == "__main__":
    main()
