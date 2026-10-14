#!/usr/bin/env python
"""Prepare HAM10000 metadata and leakage-safe splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ham10000 import load_ham10000
from src.data.splits import make_lesion_splits
from src.data.targets import get_target_spec, prepare_task_dataframe
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/HAM10000")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", default="cancer", choices=["cancer", "melanoma", "multiclass"])
    args = parser.parse_args()

    set_seed(args.seed)
    target = get_target_spec(args.task)
    df = prepare_task_dataframe(load_ham10000(args.data_root), args.task)
    train_df, val_df, test_df = make_lesion_splits(
        df, args.out_dir, label_col=target.label_column, seed=args.seed
    )
    print(f"Prepared {len(df)} images: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")


if __name__ == "__main__":
    main()
