"""HAM10000 metadata loading and path resolution."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROCESSED_COLUMNS = [
    "image_id",
    "image_path",
    "lesion_id",
    "binary_label",
    "multiclass_label",
    "dx",
    "dx_type",
    "age",
    "sex",
    "localization",
]


def load_ham10000(data_root: str | Path, processed_dir: str | Path = "data/processed") -> pd.DataFrame:
    """Load HAM10000 metadata, resolve image paths, add labels, and save a processed CSV."""
    data_root = Path(data_root)
    metadata_path = data_root / "HAM10000_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Could not find HAM10000 metadata at {metadata_path}. "
            "Expected data_root/HAM10000_metadata.csv."
        )

    df = pd.read_csv(metadata_path)
    required = {"lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"HAM10000 metadata is missing required columns: {missing}")

    df = df.copy()
    df["image_path"] = df["image_id"].map(lambda image_id: str(_resolve_image_path(data_root, image_id)))
    df["binary_label"] = (df["dx"] == "mel").astype(int)
    df["multiclass_label"] = df["dx"].astype(str)
    df = df[PROCESSED_COLUMNS]

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_dir / "ham10000_metadata_processed.csv", index=False)
    return df


def _resolve_image_path(data_root: Path, image_id: str) -> Path:
    filename = f"{image_id}.jpg"
    candidates = [
        data_root / "HAM10000_images_part_1" / filename,
        data_root / "HAM10000_images_part_2" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {filename} in HAM10000_images_part_1 or HAM10000_images_part_2 under {data_root}."
    )
