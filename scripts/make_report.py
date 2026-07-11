#!/usr/bin/env python
"""Print a compact summary of saved experiment results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()
    leaderboard_path = Path(args.out_dir) / "reports" / "leaderboard.csv"
    if not leaderboard_path.exists():
        raise FileNotFoundError(f"No leaderboard found at {leaderboard_path}. Run scripts/train_all.py first.")
    leaderboard = pd.read_csv(leaderboard_path)
    print(leaderboard.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
