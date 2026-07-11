#!/usr/bin/env python
"""Extract or reuse cached ABCDE features and medical image embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ham10000 import load_ham10000
from src.features.abcde import extract_abcde_dataframe
from src.features.cache import (
    abcde_cache_path,
    embedding_cache_path,
    load_abcde_cache,
    load_embedding_cache,
    save_abcde_cache,
    save_embedding_cache,
)
from src.features.embeddings import extract_embeddings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/HAM10000")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--embedding_model_name", default="yyupenn/whylesionclip")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force_recompute_abcde", action="store_true")
    parser.add_argument("--force_recompute_embeddings", action="store_true")
    args = parser.parse_args()

    df = load_ham10000(args.data_root)
    abcde_path = abcde_cache_path(args.out_dir)
    if args.force_recompute_abcde or load_abcde_cache(abcde_path) is None:
        abcde_df = extract_abcde_dataframe(df["image_path"], df["image_id"])
        save_abcde_cache(abcde_df, abcde_path)
        print(f"Saved ABCDE features to {abcde_path}")
    else:
        print(f"Using cached ABCDE features at {abcde_path}")

    emb_path = embedding_cache_path(args.out_dir, args.embedding_model_name)
    if args.force_recompute_embeddings or load_embedding_cache(emb_path) is None:
        embeddings = extract_embeddings(
            df["image_path"],
            model_name=args.embedding_model_name,
            batch_size=args.batch_size,
            device=args.device,
        )
        save_embedding_cache(
            emb_path,
            embeddings,
            image_ids=df["image_id"].astype(str).tolist(),
            image_paths=df["image_path"].astype(str).tolist(),
            model_name=args.embedding_model_name,
        )
        print(f"Saved embeddings to {emb_path}")
    else:
        print(f"Using cached embeddings at {emb_path}")


if __name__ == "__main__":
    main()
