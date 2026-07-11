#!/usr/bin/env python
"""One-command ABCDE-Tree pipeline for HAM10000."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ham10000 import load_ham10000
from src.data.splits import make_lesion_splits
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
from src.features.fusion import build_feature_sets
from src.training.train_cnn import train_cnn_baseline
from src.training.train_raw_image import train_raw_image_models
from src.training.train_sequence import train_sequence_models
from src.training.train_tabular import train_all_tabular
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ABCDE-Tree experiments on HAM10000.")
    parser.add_argument("--dataset", default="ham10000", choices=["ham10000"])
    parser.add_argument("--data_root", default="data/HAM10000")
    parser.add_argument(
        "--embedding_model_name",
        default="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    )
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--task", default="binary", choices=["binary", "multiclass"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--pca_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force_recompute_abcde", action="store_true")
    parser.add_argument("--force_recompute_embeddings", action="store_true")
    parser.add_argument("--skip_tabular", action="store_true", help="Reuse existing tabular metrics/leaderboard.")
    parser.add_argument("--tabular_models", nargs="+", default=None, help="Limit tabular training to selected models.")
    parser.add_argument("--append_results", action="store_true", help="Merge newly trained rows with existing reports.")
    parser.add_argument("--skip_sequence_models", action="store_true")
    parser.add_argument(
        "--sequence_feature_sets",
        nargs="+",
        default=["abcde_embeddings"],
        help="Prepared feature sets used by RNN/LSTM/Transformer baselines.",
    )
    parser.add_argument("--sequence_epochs", type=int, default=25)
    parser.add_argument("--sequence_batch_size", type=int, default=128)
    parser.add_argument("--sequence_lr", type=float, default=1e-3)
    parser.add_argument("--sequence_patience", type=int, default=5)
    parser.add_argument("--skip_raw_image_models", action="store_true")
    parser.add_argument(
        "--raw_image_models",
        nargs="+",
        default=["raw_rnn", "raw_lstm", "raw_transformer", "vit"],
        help="Raw pixel neural models to train.",
    )
    parser.add_argument("--raw_image_size", type=int, default=64)
    parser.add_argument("--raw_image_epochs", type=int, default=8)
    parser.add_argument("--raw_image_batch_size", type=int, default=64)
    parser.add_argument("--raw_image_lr", type=float, default=1e-3)
    parser.add_argument("--raw_image_patience", type=int, default=3)
    parser.add_argument("--run_cnn", action="store_true", help="Deprecated: CNN now runs by default unless skipped.")
    parser.add_argument("--skip_cnn", action="store_true")
    parser.add_argument("--cnn_epochs", type=int, default=5)
    parser.add_argument("--cnn_pretrained", action="store_true", help="Download/use pretrained ResNet18 weights.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    for subdir in ["embeddings", "features", "models", "reports", "splits", "predictions", "figures"]:
        ensure_dir(out_dir / subdir)

    df = load_ham10000(args.data_root)
    df["image_id"] = df["image_id"].astype(str)
    train_df, val_df, test_df = make_lesion_splits(df, out_dir, seed=args.seed)

    abcde_df = _get_abcde_features(df, out_dir, args.force_recompute_abcde)
    embedding_cache = _get_embeddings(
        df,
        out_dir,
        args.embedding_model_name,
        args.batch_size,
        args.device,
        args.force_recompute_embeddings,
    )

    feature_sets = build_feature_sets(
        train_df,
        val_df,
        test_df,
        abcde_df,
        embedding_cache,
        out_dir=out_dir,
        pca_dim=args.pca_dim,
    )
    existing_metrics = {}
    existing_leaderboard = pd.DataFrame()
    if args.append_results:
        existing_metrics, existing_leaderboard = _load_existing_results(out_dir)

    if args.skip_tabular:
        metrics, leaderboard = _load_existing_results(out_dir)
    else:
        metrics, leaderboard = train_all_tabular(
            feature_sets,
            train_df,
            val_df,
            test_df,
            out_dir=out_dir,
            task=args.task,
            seed=args.seed,
            model_names=args.tabular_models,
        )
        if args.append_results:
            metrics = _merge_metrics(existing_metrics, metrics)
            leaderboard = _merge_leaderboards(existing_leaderboard, leaderboard)
            leaderboard.to_csv(out_dir / "reports" / "leaderboard.csv", index=False)
            write_json(metrics, out_dir / "reports" / "metrics.json")

    if not args.skip_sequence_models:
        sequence_metrics, sequence_leaderboard = train_sequence_models(
            feature_sets,
            train_df,
            val_df,
            test_df,
            out_dir=out_dir,
            task=args.task,
            seed=args.seed,
            feature_set_names=args.sequence_feature_sets,
            epochs=args.sequence_epochs,
            batch_size=args.sequence_batch_size,
            lr=args.sequence_lr,
            patience=args.sequence_patience,
            device=args.device,
        )
        for feature_name, feature_metrics in sequence_metrics.items():
            metrics.setdefault(feature_name, {}).update(feature_metrics)
        leaderboard = _merge_leaderboards(leaderboard, sequence_leaderboard)
        leaderboard.to_csv(out_dir / "reports" / "leaderboard.csv", index=False)
        write_json(metrics, out_dir / "reports" / "metrics.json")

    if not args.skip_raw_image_models:
        raw_metrics, raw_leaderboard = train_raw_image_models(
            train_df,
            val_df,
            test_df,
            out_dir=out_dir,
            task=args.task,
            seed=args.seed,
            models=args.raw_image_models,
            image_size=args.raw_image_size,
            epochs=args.raw_image_epochs,
            batch_size=args.raw_image_batch_size,
            lr=args.raw_image_lr,
            patience=args.raw_image_patience,
            device=args.device,
        )
        metrics.setdefault("pixels", {}).update(raw_metrics)
        leaderboard = _merge_leaderboards(leaderboard, raw_leaderboard)
        leaderboard.to_csv(out_dir / "reports" / "leaderboard.csv", index=False)
        write_json(metrics, out_dir / "reports" / "metrics.json")

    if not args.skip_cnn:
        cnn_metrics = train_cnn_baseline(
            train_df,
            val_df,
            test_df,
            out_dir=out_dir,
            batch_size=args.batch_size,
            epochs=args.cnn_epochs,
            seed=args.seed,
            pretrained=args.cnn_pretrained,
        )
        metrics.setdefault("pixels", {})["cnn_resnet18"] = cnn_metrics
        write_json(metrics, out_dir / "reports" / "metrics.json")
        cnn_row = {
            "feature_set": "pixels",
            "model": "cnn_resnet18",
            **{key: cnn_metrics["test"].get(key) for key in leaderboard.columns if key not in {"feature_set", "model"}},
        }
        leaderboard = _merge_leaderboards(leaderboard, pd.DataFrame([cnn_row]))
        leaderboard.to_csv(out_dir / "reports" / "leaderboard.csv", index=False)

    print(f"Done. Leaderboard: {out_dir / 'reports' / 'leaderboard.csv'}")


def _merge_leaderboards(*leaderboards: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([df for df in leaderboards if not df.empty], ignore_index=True, sort=False)
    if {"feature_set", "model"} <= set(combined.columns):
        combined = combined.drop_duplicates(subset=["feature_set", "model"], keep="last")
    sort_columns = [col for col in ["ROC-AUC", "macro ROC-AUC", "balanced_accuracy"] if col in combined.columns]
    if sort_columns:
        combined = combined.sort_values(by=sort_columns, ascending=False, na_position="last")
    return combined


def _load_existing_results(out_dir: Path) -> tuple[dict, pd.DataFrame]:
    metrics_path = out_dir / "reports" / "metrics.json"
    leaderboard_path = out_dir / "reports" / "leaderboard.csv"
    if not metrics_path.exists() or not leaderboard_path.exists():
        raise FileNotFoundError(
            "--skip_tabular requires existing outputs/reports/metrics.json and outputs/reports/leaderboard.csv."
        )
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    leaderboard = pd.read_csv(leaderboard_path)
    return metrics, leaderboard


def _merge_metrics(base: dict, new: dict) -> dict:
    merged = dict(base)
    for feature_name, feature_metrics in new.items():
        if isinstance(feature_metrics, dict) and isinstance(merged.get(feature_name), dict):
            merged[feature_name].update(feature_metrics)
        else:
            merged[feature_name] = feature_metrics
    return merged


def _get_abcde_features(df: pd.DataFrame, out_dir: Path, force: bool) -> pd.DataFrame:
    path = abcde_cache_path(out_dir)
    cached = None if force else load_abcde_cache(path)
    if cached is not None and set(cached["image_id"].astype(str)) >= set(df["image_id"].astype(str)):
        return cached
    abcde_df = extract_abcde_dataframe(df["image_path"], df["image_id"])
    save_abcde_cache(abcde_df, path)
    return abcde_df


def _get_embeddings(
    df: pd.DataFrame,
    out_dir: Path,
    model_name: str,
    batch_size: int,
    device: str | None,
    force: bool,
) -> dict:
    path = embedding_cache_path(out_dir, model_name)
    cached = None if force else load_embedding_cache(path)
    if cached is not None and set(cached["image_ids"].astype(str)) >= set(df["image_id"].astype(str)):
        return cached
    embeddings = extract_embeddings(df["image_path"], model_name=model_name, batch_size=batch_size, device=device)
    save_embedding_cache(
        path,
        embeddings,
        image_ids=df["image_id"].astype(str).tolist(),
        image_paths=df["image_path"].astype(str).tolist(),
        model_name=model_name,
    )
    return load_embedding_cache(path)


if __name__ == "__main__":
    main()
