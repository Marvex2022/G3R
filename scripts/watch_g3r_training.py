#!/usr/bin/env python3
"""Hourly health check and automatic restart for the four-GPU G3R run."""

from __future__ import annotations

import argparse
import csv
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "runs/g3r_pandaset_4gpu"
METRICS = OUTPUT / "metrics.csv"
LATEST = OUTPUT / "latest.pt"
FALLBACK = PROJECT / "runs/g3r_pandaset/checkpoint_0025.pt"
TRAIN_LOG = OUTPUT / "train.log"
WATCH_LOG = OUTPUT / "watchdog.log"
PID_FILE = OUTPUT / "watchdog.pid"
LOCK_FILE = OUTPUT / ".watchdog.lock"
PAUSE_FILE = OUTPUT / ".watchdog_paused"
PYTHON = Path("/home/user/miniconda3/envs/g3r-repro/bin/python")
TORCHRUN = Path("/home/user/miniconda3/envs/g3r-repro/bin/torchrun")
INTERVAL_SECONDS = 3600
TOTAL_ITERATIONS = 1000


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with WATCH_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def latest_metric() -> dict[str, str] | None:
    if not METRICS.exists():
        return None
    last = None
    try:
        with METRICS.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                last = row
    except (OSError, csv.Error):
        return None
    return last


def training_pids() -> list[int]:
    pids: list[int] = []
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            cmd = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "train.py" in cmd and "runs/g3r_pandaset_4gpu" in cmd and "watch_g3r_training.py" not in cmd:
            pids.append(int(proc_dir.name))
    return sorted(pids)


def gpu_snapshot() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            cwd=PROJECT,
            text=True,
            capture_output=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "GPU status unavailable"
    selected = []
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if fields and fields[0] in {"1", "2", "5", "7"}:
            selected.append(f"GPU{fields[0]}={fields[1]}MiB/{fields[2]}%")
    return " ".join(selected)


def restart_training(checkpoint: Path) -> int:
    env = os.environ.copy()
    env.update(
        {
            # Cron uses a minimal PATH. gsplat's JIT loader calls the Ninja
            # executable directly, which is installed in the base Conda bin.
            "PATH": (
                "/home/user/miniconda3/envs/g3r-repro/bin:"
                "/home/user/miniconda3/bin:"
                + env.get("PATH", "/usr/local/bin:/usr/bin:/bin")
            ),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "G3R_DIST_BACKEND": "gloo",
            "CUDA_VISIBLE_DEVICES": "1,2,5,7",
        }
    )
    command = [
        str(TORCHRUN),
        "--standalone",
        "--nproc-per-node=4",
        "train.py",
        "--config",
        "configs/pandaset_paper.yaml",
        "--output",
        "runs/g3r_pandaset_4gpu",
        "--device",
        "cuda",
        "--resume",
        str(checkpoint),
    ]
    with TRAIN_LOG.open("a", encoding="utf-8") as train_log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=train_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return process.pid


def check_once() -> bool:
    if PAUSE_FILE.exists():
        log(f"watchdog paused by {PAUSE_FILE}")
        return True
    metric = latest_metric()
    iteration = int(metric["iteration"]) if metric and metric.get("iteration", "").isdigit() else 0
    pids = training_pids()
    metric_age = int(time.time() - METRICS.stat().st_mtime) if METRICS.exists() else -1
    if pids:
        mse = metric.get("mse", "?") if metric else "?"
        psnr = metric.get("psnr_db", "?") if metric else "?"
        log(
            f"healthy: iteration={iteration} mse={mse} psnr={psnr} "
            f"metric_age={metric_age}s pids={pids} {gpu_snapshot()}"
        )
        return True

    if iteration >= TOTAL_ITERATIONS:
        log(f"training complete at iteration={iteration}; watchdog exiting")
        return False

    checkpoint = LATEST if LATEST.exists() else FALLBACK
    if not checkpoint.exists():
        log("training stopped, but neither latest.pt nor checkpoint_0025.pt exists; will retry next hour")
        return True

    pid = restart_training(checkpoint)
    log(f"training was stopped; restarted pid={pid} from {checkpoint}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Run one health check and exit")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK_FILE.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another G3R watchdog is already running.", file=sys.stderr)
        return 1

    PID_FILE.write_text(f"{os.getpid()}\n", encoding="utf-8")
    running = True

    def stop(_signum, _frame) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if args.once:
        log("scheduled health check started")
        try:
            check_once()
        finally:
            PID_FILE.unlink(missing_ok=True)
        return 0

    log(f"watchdog started; interval={INTERVAL_SECONDS}s")
    try:
        while running and check_once():
            deadline = time.monotonic() + INTERVAL_SECONDS
            while running and time.monotonic() < deadline:
                time.sleep(min(30, deadline - time.monotonic()))
    finally:
        PID_FILE.unlink(missing_ok=True)
        log("watchdog stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
