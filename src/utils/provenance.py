"""Reproducibility and compute-provenance helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def collect_run_manifest(argv: list[str], out_dir: str | Path, target: dict) -> dict:
    out_dir = Path(out_dir)
    commit = _git_output(["rev-parse", "HEAD"]) or "unavailable"
    status = _git_output(["status", "--porcelain"])
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *argv]),
        "code_commit": commit,
        "git_dirty": bool(status),
        "target": target,
        "hardware": _hardware_info(),
        "software": _software_info(),
        "split_files": _split_file_info(out_dir / "splits"),
    }


def _git_output(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _hardware_info() -> dict:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "ram_gib": _ram_gib(),
        "cuda_available": False,
        "gpu": None,
    }
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except Exception:
        pass
    return info


def _ram_gib() -> float | None:
    try:
        fields = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
        total_kib = int(next(line for line in fields if line.startswith("MemTotal:")).split()[1])
        return round(total_kib / 1024**2, 2)
    except Exception:
        return None


def _software_info() -> dict:
    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "torch",
        "torchvision",
        "xgboost",
        "lightgbm",
        "catboost",
        "transformers",
        "open-clip-torch",
    ]
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _split_file_info(split_dir: Path) -> dict:
    result = {}
    for name in ("train", "val", "test"):
        path = split_dir / f"{name}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=["image_id", "lesion_id"])
        result[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "images": int(frame["image_id"].nunique()),
            "unique_lesion_id": int(frame["lesion_id"].nunique()),
        }
    names = list(result)
    overlaps = {}
    frames = {
        name: set(pd.read_csv(split_dir / f"{name}.csv", usecols=["lesion_id"])["lesion_id"])
        for name in names
    }
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlaps[f"{left}_{right}"] = len(frames[left] & frames[right])
    result["lesion_id_overlap_counts"] = overlaps
    return result
