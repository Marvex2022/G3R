#!/usr/bin/env python3
"""Convert raw PandaSet sequences into G3R scene manifests.

The adapter separates points inside non-stationary PandaSet cuboids from the
background, aggregates them in each actor's local frame, and writes per-camera
actor-to-world transforms. Duplicate front-lidar cuboids (sensor_id=1) are
ignored, matching the PandaSet parser used by NeuRAD.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import zlib

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g3r.config import load_config


def pose_to_c2w(pose: dict) -> np.ndarray:
    q = pose["heading"]
    # scipy uses xyzw; PandaSet stores wxyz.
    rotation = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    position = pose["position"]
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = [position["x"], position["y"], position["z"]]
    return c2w


def available_sequences(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 3)


def sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(len(points), size=max_points, replace=False)]
    return points


def yaw_transform(position: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    transform[:3, 3] = position
    return transform


def dynamic_cuboids(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one canonical cuboid per moving actor.

    PandaSet sometimes provides sibling cuboids for the 360 and front-facing
    lidars. sensor_id=1 is the duplicate front-lidar annotation and is skipped.
    """
    required = {
        "uuid", "yaw", "stationary", "position.x", "position.y", "position.z",
        "dimensions.x", "dimensions.y", "dimensions.z", "cuboids.sensor_id",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Cuboid annotation missing columns: {sorted(missing)}")
    selected = frame[(~frame["stationary"].astype(bool)) & (frame["cuboids.sensor_id"] != 1)]
    return selected.drop_duplicates(subset="uuid", keep="first")


def split_dynamic_scaffolds(
    sequence_root: Path,
    background_max_points: int,
    actor_max_points: int,
    actor_min_points: int,
    cuboid_margin: float,
    seed: int,
    include_dynamic: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[int, dict[str, np.ndarray]], dict[str, str]]:
    lidar_files = sorted((sequence_root / "lidar").glob("*.pkl.gz"))
    if not lidar_files:
        raise FileNotFoundError(f"No LiDAR frames found under {sequence_root / 'lidar'}")

    background_chunks: list[np.ndarray] = []
    actor_chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    actor_poses: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    actor_labels: dict[str, str] = {}
    annotation_root = sequence_root / "annotations" / "cuboids"

    for lidar_path in lidar_files:
        frame_id = int(lidar_path.name.split(".", 1)[0])
        lidar = pd.read_pickle(lidar_path)
        points = lidar[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=True)
        points = points[np.isfinite(points).all(axis=1)]
        background_mask = np.ones(len(points), dtype=bool)

        if include_dynamic:
            annotation_path = annotation_root / lidar_path.name
            if not annotation_path.is_file():
                raise FileNotFoundError(annotation_path)
            cuboids = dynamic_cuboids(pd.read_pickle(annotation_path))
            for _, cuboid in cuboids.iterrows():
                actor_id = str(cuboid["uuid"]).strip()
                position = np.array(
                    [cuboid["position.x"], cuboid["position.y"], cuboid["position.z"]],
                    dtype=np.float64,
                )
                dimensions = np.array(
                    [cuboid["dimensions.x"], cuboid["dimensions.y"], cuboid["dimensions.z"]],
                    dtype=np.float64,
                )
                transform = yaw_transform(position, float(cuboid["yaw"]))
                rotation = transform[:3, :3]
                half_extent = dimensions / 2.0 + cuboid_margin
                # Cheap world-axis AABB rejection keeps the exact oriented-box
                # test proportional to nearby points instead of all ~170k points.
                world_extent = np.abs(rotation) @ half_extent
                candidate = np.all(np.abs(points - position) <= world_extent, axis=1)
                candidate_indices = np.flatnonzero(candidate)
                if len(candidate_indices):
                    local = (points[candidate_indices].astype(np.float64) - position) @ rotation
                    inside_local = np.all(np.abs(local) <= half_extent, axis=1)
                    inside_indices = candidate_indices[inside_local]
                    if len(inside_indices):
                        actor_chunks[actor_id].append(local[inside_local].astype(np.float32))
                        background_mask[inside_indices] = False
                actor_poses[frame_id][actor_id] = transform
                actor_labels[actor_id] = str(cuboid.get("label", "Unknown"))

        background_chunks.append(points[background_mask])

    background = sample_points(np.concatenate(background_chunks, axis=0), background_max_points, seed)
    actors: dict[str, np.ndarray] = {}
    for actor_id, chunks in actor_chunks.items():
        points = np.concatenate(chunks, axis=0)
        if len(points) < actor_min_points:
            continue
        actor_seed = seed + zlib.crc32(actor_id.encode("utf-8"))
        actors[actor_id] = sample_points(points, actor_max_points, actor_seed)

    retained = set(actors)
    poses = {
        frame_id: {actor_id: transform for actor_id, transform in transforms.items() if actor_id in retained}
        for frame_id, transforms in actor_poses.items()
    }
    labels = {actor_id: actor_labels[actor_id] for actor_id in retained}
    return background, actors, poses, labels


def third_neighbor_scale(points: np.ndarray) -> np.ndarray:
    if len(points) < 4:
        return np.full(len(points), 0.05, dtype=np.float32)
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=4, workers=-1)
    return np.maximum(distances[:, 3], 1e-4).astype(np.float32)


def camera_views(sequence_root: Path, camera: str) -> list[dict]:
    camera_root = sequence_root / "camera" / camera
    with open(camera_root / "intrinsics.json", "r", encoding="utf-8") as f:
        intr = json.load(f)
    with open(camera_root / "poses.json", "r", encoding="utf-8") as f:
        poses = json.load(f)
    K = [[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]]
    images = {int(path.stem): path for path in camera_root.glob("*.jpg")}
    views = []
    for frame_id, pose in enumerate(poses):
        image = images.get(frame_id)
        if image is None or not image.is_file():
            raise FileNotFoundError(f"No image for frame {frame_id} under {camera_root}")
        w2c = np.linalg.inv(pose_to_c2w(pose))
        views.append({
            "image": str(image.resolve()),
            "K": K,
            "w2c": w2c.tolist(),
            "width": 1920,
            "height": 1080,
            "frame_id": frame_id,
            "split": "source" if frame_id % 2 == 0 else "target",
        })
    return views


def convert_sequence(
    data_root: Path,
    prepared_root: Path,
    sequence: str,
    camera: str,
    max_points: int,
    seed: int,
    dynamic_actors: bool,
    actor_max_points: int,
    actor_min_points: int,
    cuboid_margin: float,
) -> Path:
    source = data_root / sequence
    destination = prepared_root / sequence
    destination.mkdir(parents=True, exist_ok=True)
    # Validate camera assets before doing the expensive LiDAR aggregation.
    views = camera_views(source, camera)
    points, actors, actor_poses, actor_labels = split_dynamic_scaffolds(
        source,
        background_max_points=max_points,
        actor_max_points=actor_max_points,
        actor_min_points=actor_min_points,
        cuboid_margin=cuboid_margin,
        seed=seed,
        include_dynamic=dynamic_actors,
    )
    # Precompute this once; otherwise every scene initialization repeats an
    # expensive nearest-neighbor search over the full scaffold.
    scale = third_neighbor_scale(points)
    np.savez(destination / "background.npz", xyz=points, scale=scale)
    actor_root = destination / "actors"
    actor_root.mkdir(exist_ok=True)
    actor_paths = {}
    for actor_id, actor_points in actors.items():
        actor_path = actor_root / f"actor_{actor_id}.npz"
        np.savez(actor_path, xyz=actor_points, scale=third_neighbor_scale(actor_points))
        actor_paths[actor_id] = str(actor_path.relative_to(destination))
    for view in views:
        view["actor_to_world"] = {
            actor_id: transform.tolist()
            for actor_id, transform in actor_poses.get(view["frame_id"], {}).items()
        }
    manifest = {
        "name": f"pandaset_{sequence}",
        "components": {"background": "background.npz", "actors": actor_paths},
        "views": views,
        "metadata": {
            "source_dataset": "PandaSet",
            "sequence": sequence,
            "adapter_mode": "dynamic_actor_local" if dynamic_actors else "static_background_baseline",
            "actor_labels": actor_labels,
            "cuboid_margin": cuboid_margin,
            "background_max_points": max_points,
            "actor_max_points": actor_max_points,
            "actor_min_points": actor_min_points,
        },
    }
    scene_path = destination / "scene.json"
    with open(scene_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    actor_point_count = sum(len(x) for x in actors.values())
    print(
        f"{sequence}: {len(points):,} background points, {len(actors)} actors/"
        f"{actor_point_count:,} actor points, {len(manifest['views'])} views -> {scene_path}"
    )
    return scene_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pandaset_paper.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--prepared-root")
    parser.add_argument("--sequences", nargs="+", help="Sequence IDs; defaults to all available sequences")
    parser.add_argument("--camera", default="front_camera")
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--static-only", action="store_true", help="Do not extract dynamic actor-local scaffolds")
    parser.add_argument("--actor-max-points", type=int)
    parser.add_argument("--actor-min-points", type=int)
    parser.add_argument("--cuboid-margin", type=float)
    parser.add_argument("--reuse-existing", action="store_true", help="Keep already converted scene/background files")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config.get("paths", {})
    data_root_value = args.data_root or paths.get("pandaset_root")
    prepared_root_value = args.prepared_root or paths.get("prepared_root")
    if not data_root_value or not prepared_root_value:
        parser.error("data/prepared roots must be given by CLI or config paths")
    data_root = Path(data_root_value).expanduser().resolve()
    prepared_root = Path(prepared_root_value).expanduser().resolve()
    if not data_root.is_dir():
        parser.error(f"PandaSet root does not exist: {data_root}")
    sequences = args.sequences or available_sequences(data_root)
    max_points = args.max_points or int(config["training"]["point_limit"])
    seed = args.seed if args.seed is not None else int(config["training"].get("seed", 0))
    pcfg = config.get("preprocessing", {})
    dynamic_actors = bool(pcfg.get("dynamic_actors", True)) and not args.static_only
    actor_max_points = args.actor_max_points or int(pcfg.get("actor_max_points", 50_000))
    actor_min_points = args.actor_min_points or int(pcfg.get("actor_min_points", 20))
    cuboid_margin = args.cuboid_margin if args.cuboid_margin is not None else float(pcfg.get("cuboid_margin", 0.05))
    prepared_root.mkdir(parents=True, exist_ok=True)

    scene_paths = []
    for sequence in sequences:
        existing = prepared_root / sequence / "scene.json"
        background = prepared_root / sequence / "background.npz"
        expected_mode = "dynamic_actor_local" if dynamic_actors else "static_background_baseline"
        existing_metadata = {}
        if existing.is_file():
            with open(existing, "r", encoding="utf-8") as f:
                existing_metadata = json.load(f).get("metadata", {})
        reusable = (
            existing_metadata.get("adapter_mode") == expected_mode
            and existing_metadata.get("background_max_points") == max_points
            and existing_metadata.get("actor_max_points") == actor_max_points
            and existing_metadata.get("actor_min_points") == actor_min_points
            and existing_metadata.get("cuboid_margin") == cuboid_margin
        )
        if args.reuse_existing and existing.is_file() and background.is_file() and reusable:
            print(f"{sequence}: reuse {existing}")
            scene_paths.append(existing)
        else:
            try:
                scene_paths.append(
                    convert_sequence(
                        data_root, prepared_root, sequence, args.camera, max_points, seed,
                        dynamic_actors, actor_max_points, actor_min_points, cuboid_margin,
                    )
                )
            except FileNotFoundError as error:
                if args.sequences:
                    raise
                print(f"{sequence}: skip incomplete sequence ({error})", file=sys.stderr)
    scene_list_path = Path(paths.get("scenes", prepared_root / "train_scenes.json")).resolve()
    scene_list_path.parent.mkdir(parents=True, exist_ok=True)
    scene_list = {"scenes": [str(path.relative_to(scene_list_path.parent)) for path in scene_paths]}
    with open(scene_list_path, "w", encoding="utf-8") as f:
        json.dump(scene_list, f, indent=2)
    print(f"scene list -> {scene_list_path}")


if __name__ == "__main__":
    main()
