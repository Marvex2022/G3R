#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import torch
import torch.distributed as dist

from g3r.config import load_config
from g3r.data import load_manifest_list
from g3r.engine import train_g3r


def _map_rank_to_cuda_device(
    local_rank: int,
    world_size: int,
    visible_device_count: int,
    processes_per_gpu: int,
) -> int:
    """Map adjacent local workers onto the same visible CUDA device."""
    if visible_device_count <= 0:
        raise ValueError("No visible CUDA devices")
    if processes_per_gpu <= 0:
        raise ValueError("G3R_PROCESSES_PER_GPU must be positive")
    expected_world_size = visible_device_count * processes_per_gpu
    if world_size != expected_world_size:
        raise ValueError(
            f"WORLD_SIZE={world_size} does not match {visible_device_count} visible GPUs "
            f"x G3R_PROCESSES_PER_GPU={processes_per_gpu} ({expected_world_size})"
        )
    if not 0 <= local_rank < world_size:
        raise ValueError(f"LOCAL_RANK={local_rank} is outside WORLD_SIZE={world_size}")
    return local_rank // processes_per_gpu


def _initialize_gsplat_backend(distributed: bool) -> None:
    """Load gsplat serially because its JIT loader removes the shared lock file."""
    if not distributed:
        from gsplat.cuda._backend import _C  # noqa: F401
        return

    rank = dist.get_rank()
    for owner in range(dist.get_world_size()):
        if rank == owner:
            from gsplat.cuda._backend import _C  # noqa: F401
        # Do not let another rank enter gsplat's unsafe JIT loader concurrently.
        dist.barrier()


def main():
    p = argparse.ArgumentParser("Train G3R learned reconstruction optimizer")
    p.add_argument("--config", required=True)
    p.add_argument("--scenes", help="JSON scene list; defaults to paths.scenes in the config")
    p.add_argument("--output", help="Checkpoint directory; defaults to paths.output in the config")
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", help="Resume modules, optimizer, iteration, and global step from a checkpoint")
    args = p.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gsplat/TorchSparse training")

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        if not args.device.startswith("cuda"):
            raise ValueError("Multi-GPU training requires --device cuda")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        processes_per_gpu = int(os.environ.get("G3R_PROCESSES_PER_GPU", "1"))
        device_index = _map_rank_to_cuda_device(
            local_rank,
            world_size,
            torch.cuda.device_count(),
            processes_per_gpu,
        )
        torch.cuda.set_device(device_index)
        device = f"cuda:{device_index}"
        backend = os.environ.get("G3R_DIST_BACKEND", "nccl").lower()
        if backend == "nccl":
            dist.init_process_group(backend=backend, device_id=torch.device(device))
        elif backend == "gloo":
            dist.init_process_group(backend=backend)
        else:
            raise ValueError(f"Unsupported G3R_DIST_BACKEND={backend!r}; use nccl or gloo")
    else:
        device = args.device
    _initialize_gsplat_backend(distributed)
    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    scenes_path = args.scenes or paths.get("scenes")
    output_path = args.output or paths.get("output")
    if not scenes_path:
        p.error("--scenes is required unless paths.scenes is set in the config")
    if not output_path:
        p.error("--output is required unless paths.output is set in the config")
    scenes = load_manifest_list(scenes_path)
    if not scenes:
        raise ValueError("No scenes found")
    try:
        train_g3r(cfg, scenes, output_path, device, resume=args.resume)
    finally:
        if distributed:
            try:
                dist.destroy_process_group()
            except dist.DistError:
                # Preserve the original NCCL/CUDA exception during failed startup.
                pass


if __name__ == "__main__":
    main()
