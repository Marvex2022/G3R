#!/usr/bin/env python3
"""Render a reconstructed neural state saved by a future extension.

For now, `infer.py --render-targets` is the recommended path because the exported
NPZ stores explicit per-component Gaussians while dynamic actor transforms are
view-dependent and need the scene manifest to compose correctly.
"""

if __name__ == "__main__":
    raise SystemExit("Use: python infer.py ... --render-targets")
