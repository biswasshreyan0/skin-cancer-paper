"""Cache helpers for features and embeddings."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def abcde_cache_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / "features" / "abcd_features.csv"


def embedding_cache_path(out_dir: str | Path, model_name: str) -> Path:
    return Path(out_dir) / "embeddings" / f"{safe_name(model_name)}.npz"


def load_abcde_cache(path: str | Path) -> pd.DataFrame | None:
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return None


def save_abcde_cache(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_embedding_cache(
    path: str | Path,
    embeddings: np.ndarray,
    image_ids: list[str],
    image_paths: list[str],
    model_name: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=embeddings,
        image_ids=np.asarray(image_ids),
        image_paths=np.asarray(image_paths),
        model_name=np.asarray(model_name),
        embedding_dim=np.asarray(embeddings.shape[1]),
    )


def load_embedding_cache(path: str | Path) -> dict[str, np.ndarray] | None:
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}
