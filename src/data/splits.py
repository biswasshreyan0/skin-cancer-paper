"""Leakage-safe grouped splitting by lesion_id."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit


def make_lesion_splits(
    df: pd.DataFrame,
    out_dir: str | Path,
    label_col: str = "binary_label",
    group_col: str = "lesion_id",
    seed: int = 42,
    train_size: float = 0.80,
    val_size: float = 0.10,
    test_size: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows while keeping every lesion_id in exactly one split."""
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {total:.4f}.")
    if group_col not in df or label_col not in df:
        raise ValueError(f"DataFrame must contain {group_col!r} and {label_col!r}.")

    group_labels = df.groupby(group_col)[label_col].agg(lambda s: s.value_counts().idxmax()).reset_index()
    train_groups, temp_groups = _stratified_group_sample(
        group_labels[group_col].to_numpy(),
        group_labels[label_col].to_numpy(),
        train_size=train_size,
        seed=seed,
    )

    temp = group_labels[group_labels[group_col].isin(temp_groups)]
    val_fraction_of_temp = val_size / (val_size + test_size)
    val_groups, test_groups = _stratified_group_sample(
        temp[group_col].to_numpy(),
        temp[label_col].to_numpy(),
        train_size=val_fraction_of_temp,
        seed=seed + 1,
    )

    train_df = df[df[group_col].isin(train_groups)].copy()
    val_df = df[df[group_col].isin(val_groups)].copy()
    test_df = df[df[group_col].isin(test_groups)].copy()

    if not set(train_df[group_col]).isdisjoint(val_df[group_col]):
        raise RuntimeError("Leakage detected between train and validation lesion groups.")
    if not set(train_df[group_col]).isdisjoint(test_df[group_col]):
        raise RuntimeError("Leakage detected between train and test lesion groups.")
    if not set(val_df[group_col]).isdisjoint(test_df[group_col]):
        raise RuntimeError("Leakage detected between validation and test lesion groups.")

    split_dir = Path(out_dir) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(split_dir / "train.csv", index=False)
    val_df.to_csv(split_dir / "val.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)
    return train_df, val_df, test_df


def _stratified_group_sample(
    groups: np.ndarray,
    labels: np.ndarray,
    train_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return selected and held-out groups, stratifying when class counts allow it."""
    groups = np.asarray(groups)
    labels = np.asarray(labels)
    if len(groups) < 3:
        raise ValueError("Need at least three lesion groups to create train/val/test splits.")

    counts = pd.Series(labels).value_counts()
    can_stratify = len(counts) > 1 and counts.min() >= 2
    if can_stratify:
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(groups, labels))
    else:
        splitter = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(groups, groups=groups))
    return groups[train_idx], groups[test_idx]
