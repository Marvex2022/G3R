#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_pandaset import dynamic_cuboids, yaw_transform


def test_yaw_transform_roundtrip() -> None:
    transform = yaw_transform(np.array([10.0, -2.0, 1.0]), math.pi / 2.0)
    local = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, -1.0]])
    world = local @ transform[:3, :3].T + transform[:3, 3]
    recovered = (world - transform[:3, 3]) @ transform[:3, :3]
    np.testing.assert_allclose(recovered, local, atol=1e-7)


def test_dynamic_cuboid_filter() -> None:
    common = {
        "yaw": 0.0,
        "position.x": 0.0,
        "position.y": 0.0,
        "position.z": 0.0,
        "dimensions.x": 2.0,
        "dimensions.y": 4.0,
        "dimensions.z": 2.0,
    }
    frame = pd.DataFrame([
        {**common, "uuid": "moving", "stationary": False, "cuboids.sensor_id": -1},
        {**common, "uuid": "parked", "stationary": True, "cuboids.sensor_id": -1},
        {**common, "uuid": "front-duplicate", "stationary": False, "cuboids.sensor_id": 1},
    ])
    assert dynamic_cuboids(frame)["uuid"].tolist() == ["moving"]


if __name__ == "__main__":
    test_yaw_transform_roundtrip()
    test_dynamic_cuboid_filter()
    print("PandaSet adapter tests: OK")
