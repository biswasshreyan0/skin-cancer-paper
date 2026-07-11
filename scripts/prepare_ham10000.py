#!/usr/bin/env python
"""Prepare HAM10000 metadata and leakage-safe splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ham10000 import load_ham10000
from src.data.splits import make_lesion_splits
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/HAM10000")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    df = load_ham10000(args.data_root)
    train_df, val_df, test_df = make_lesion_splits(df, args.out_dir, seed=args.seed)
    print(f"Prepared {len(df)} images: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")


if __name__ == "__main__":
    main()
