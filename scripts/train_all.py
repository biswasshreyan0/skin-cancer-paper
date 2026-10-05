#!/usr/bin/env python
"""One-command ABCD/embedding pipeline for HAM10000."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ham10000 import load_ham10000
from src.data.splits import export_repository_split_manifests, make_lesion_splits
from src.data.targets import canonical_task, get_target_spec, prepare_task_dataframe
from src.features.abcde import ABCD_FEATURE_COLUMNS, extract_abcd_dataframe
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
from src.models.registry import get_model_registry
from src.reporting.table1 import write_final_table
from src.training.cross_validation import run_repeated_group_cv
from src.training.train_cnn import train_cnn_baseline
from src.training.train_raw_image import train_raw_image_models
from src.training.train_sequence import train_sequence_models
from src.training.train_tabular import train_all_tabular
from src.utils.io import ensure_dir, write_json
from src.utils.provenance import collect_run_manifest
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lesion-grouped ABCD/embedding experiments on HAM10000.")
    parser.add_argument("--dataset", default="ham10000", choices=["ham10000"])
    parser.add_argument("--data_root", default="data/HAM10000")
    parser.add_argument(
        "--embedding_model_name",
        default="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    )
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument(
        "--shared_split_dir",
        default="splits/table1",
        help="Repository-safe split manifests used by the final Table 1.",
    )
    parser.add_argument(
        "--task",
        default="cancer",
        choices=["cancer", "melanoma", "multiclass", "binary"],
        help="Prediction target. 'binary' is a deprecated alias for melanoma.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--pca_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--force_recompute_abcd",
        "--force_recompute_abcde",
        dest="force_recompute_abcd",
        action="store_true",
    )
    parser.add_argument("--force_recompute_embeddings", action="store_true")
    parser.add_argument("--skip_tabular", action="store_true", help="Reuse existing tabular metrics/leaderboard.")
    parser.add_argument("--tabular_models", nargs="+", default=None, help="Limit tabular training to selected models.")
    parser.add_argument("--append_results", action="store_true", help="Merge newly trained rows with existing reports.")
    parser.add_argument(
        "--no_resume",
        dest="resume",
        action="store_false",
        help="Recompute completed stages/jobs instead of resuming from outputs_cancer progress files.",
    )
    parser.set_defaults(resume=True)
    parser.add_argument("--skip_sequence_models", action="store_true")
    parser.add_argument(
        "--sequence_feature_sets",
        nargs="+",
        default=["abcd_embeddings"],
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
    parser.add_argument("--raw_image_size", type=int, default=224)
    parser.add_argument("--raw_image_epochs", type=int, default=8)
    parser.add_argument("--raw_image_batch_size", type=int, default=16)
    parser.add_argument("--raw_image_lr", type=float, default=1e-3)
    parser.add_argument("--raw_image_patience", type=int, default=3)
    parser.add_argument("--run_cnn", action="store_true", help="Deprecated: CNN now runs by default unless skipped.")
    parser.add_argument("--skip_cnn", action="store_true")
    parser.add_argument("--cnn_epochs", type=int, default=5)
    parser.add_argument("--cnn_pretrained", action="store_true", help="Download/use pretrained ResNet18 weights.")
    parser.add_argument(
        "--imbalance_strategy",
        default="hybrid",
        choices=["hybrid", "class_weight", "oversample", "none"],
        help="Hybrid uses class weights where supported and random oversampling for KNN.",
    )
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--cv_repeats", type=int, default=3)
    parser.add_argument("--cv_models", nargs="+", default=None)
    parser.add_argument("--skip_cross_validation", action="store_true")
    parser.add_argument(
        "--require_clean_git",
        action="store_true",
        help="Abort unless the experiment starts from a clean, committed worktree.",
    )
    return parser.parse_args()


def _has_models(metrics: dict, required: dict[str, list[str]]) -> bool:
    """Return True only when all requested model results are present."""
    return all(
        feature_name in metrics and all(model_name in metrics[feature_name] for model_name in model_names)
        for feature_name, model_names in required.items()
    )


def _write_stage_progress(out_dir: Path, stages: dict[str, str]) -> None:
    write_json(stages, out_dir / "reports" / "progress.json")


def main() -> None:
    args = parse_args()
    args.task = canonical_task(args.task)
    run_started = time.perf_counter()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    for subdir in ["embeddings", "features", "models", "reports", "splits", "predictions", "figures"]:
        ensure_dir(out_dir / subdir)

    target = get_target_spec(args.task)
    df = prepare_task_dataframe(load_ham10000(args.data_root), args.task)
    df["image_id"] = df["image_id"].astype(str)
    train_df, val_df, test_df = make_lesion_splits(
        df, out_dir, label_col=target.label_column, seed=args.seed
    )
    run_manifest = collect_run_manifest(sys.argv, out_dir, target.to_dict())
    if args.require_clean_git and run_manifest["git_dirty"]:
        raise RuntimeError(
            "The worktree is dirty. Commit the experiment code and split manifests before the paper run."
        )
    export_repository_split_manifests(
        train_df,
        val_df,
        test_df,
        args.shared_split_dir,
        label_col=target.label_column,
        target=target.to_dict(),
        seed=args.seed,
    )
    write_json(run_manifest, out_dir / "reports" / "run_manifest.json")

    abcde_df = _get_abcd_features(df, out_dir, args.force_recompute_abcd)
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
    reports_exist = (out_dir / "reports" / "metrics.json").exists() and (out_dir / "reports" / "leaderboard.csv").exists()
    if reports_exist:
        existing_metrics, existing_leaderboard = _load_existing_results(out_dir)
    progress = {}
    tabular_registry = get_model_registry(args.seed, use_class_weights=args.imbalance_strategy in {"hybrid", "class_weight"})
    requested_tabular_models = args.tabular_models or list(tabular_registry)
    tabular_complete = _has_models(
        existing_metrics,
        {feature_name: requested_tabular_models for feature_name in feature_sets},
    )

    if args.skip_tabular or (args.resume and tabular_complete):
        if not reports_exist:
            raise FileNotFoundError("Cannot resume tabular stage: metrics.json and leaderboard.csv are missing.")
        metrics, leaderboard = _load_existing_results(out_dir)
        progress["tabular"] = "resumed/skipped"
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
            imbalance_strategy=args.imbalance_strategy,
        )
        if args.append_results:
            metrics = _merge_metrics(existing_metrics, metrics)
            leaderboard = _merge_leaderboards(existing_leaderboard, leaderboard)
            leaderboard.to_csv(out_dir / "reports" / "leaderboard.csv", index=False)
            write_json(metrics, out_dir / "reports" / "metrics.json")
        progress["tabular"] = "complete"
    _write_stage_progress(out_dir, progress)

    sequence_complete = _has_models(
        metrics,
        {feature_name: ["rnn", "lstm", "transformer"] for feature_name in args.sequence_feature_sets},
    )
    if not args.skip_sequence_models and not (args.resume and sequence_complete):
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
        progress["sequence"] = "complete"
    else:
        progress["sequence"] = "resumed/skipped"
    _write_stage_progress(out_dir, progress)

    raw_feature_name = f"pixels_{args.raw_image_size}"
    raw_complete = _has_models(metrics, {raw_feature_name: args.raw_image_models})
    if not args.skip_raw_image_models and not (args.resume and raw_complete):
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
        metrics.setdefault(raw_feature_name, {}).update(raw_metrics)
        leaderboard = _merge_leaderboards(leaderboard, raw_leaderboard)
        leaderboard.to_csv(out_dir / "reports" / "leaderboard.csv", index=False)
        write_json(metrics, out_dir / "reports" / "metrics.json")
        progress["raw_image"] = "complete"
    else:
        progress["raw_image"] = "resumed/skipped"
    _write_stage_progress(out_dir, progress)

    cnn_complete = "pixels_224" in metrics and "cnn_resnet18" in metrics["pixels_224"]
    if not args.skip_cnn and target.is_binary and not (args.resume and cnn_complete):
        cnn_metrics = train_cnn_baseline(
            train_df,
            val_df,
            test_df,
            out_dir=out_dir,
            batch_size=args.batch_size,
            epochs=args.cnn_epochs,
            seed=args.seed,
            pretrained=args.cnn_pretrained,
            task=args.task,
            device=args.device,
        )
        metrics.setdefault("pixels_224", {})["cnn_resnet18"] = cnn_metrics
        write_json(metrics, out_dir / "reports" / "metrics.json")
        cnn_row = {
            "feature_set": "pixels_224",
            "model": "cnn_resnet18",
            **{key: cnn_metrics["test"].get(key) for key in leaderboard.columns if key not in {"feature_set", "model"}},
        }
        leaderboard = _merge_leaderboards(leaderboard, pd.DataFrame([cnn_row]))
        leaderboard.to_csv(out_dir / "reports" / "leaderboard.csv", index=False)
        progress["cnn"] = "complete"
    else:
        progress["cnn"] = "resumed/skipped"
    _write_stage_progress(out_dir, progress)

    cv_summary = pd.DataFrame()
    if not args.skip_cross_validation:
        development_df = pd.concat([train_df, val_df], ignore_index=True)
        _, cv_summary = run_repeated_group_cv(
            development_df,
            abcde_df,
            embedding_cache,
            out_dir=out_dir,
            task=args.task,
            pca_dim=args.pca_dim,
            seed=args.seed,
            folds=args.cv_folds,
            repeats=args.cv_repeats,
            model_names=args.cv_models or args.tabular_models,
            imbalance_strategy=args.imbalance_strategy,
        )
        progress["cross_validation"] = "complete"
    else:
        progress["cross_validation"] = "skipped"
    _write_stage_progress(out_dir, progress)

    run_manifest["total_pipeline_seconds"] = time.perf_counter() - run_started
    write_json(run_manifest, out_dir / "reports" / "run_manifest.json")
    write_final_table(leaderboard, metrics, out_dir, target, run_manifest, cv_summary=cv_summary)

    print(f"Done. Final Table 1: {out_dir / 'reports' / 'table1_final_results.csv'}")


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


def _get_abcd_features(df: pd.DataFrame, out_dir: Path, force: bool) -> pd.DataFrame:
    path = abcde_cache_path(out_dir)
    cached = None if force else load_abcde_cache(path)
    required_columns = {"image_id", *ABCD_FEATURE_COLUMNS}
    if (
        cached is not None
        and required_columns <= set(cached.columns)
        and set(cached["image_id"].astype(str)) >= set(df["image_id"].astype(str))
    ):
        return cached
    abcde_df = extract_abcd_dataframe(df["image_path"], df["image_id"])
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
