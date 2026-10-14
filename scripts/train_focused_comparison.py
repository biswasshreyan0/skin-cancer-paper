#!/usr/bin/env python
"""Focused model comparison: LR, KNN, MLP on ABCD+embeddings, plus optional CNN."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ham10000 import load_ham10000
from src.data.splits import make_lesion_splits
from src.data.targets import get_target_spec, prepare_task_dataframe
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
from src.training.evaluate import evaluate_model
from src.training.train_cnn import train_cnn_baseline
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LR, KNN, MLP, and optional CNN.")
    parser.add_argument("--data_root", default="data/HAM10000")
    parser.add_argument("--embedding_model_name", default="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--pca_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run_cnn", action="store_true")
    parser.add_argument("--cnn_epochs", type=int, default=5)
    parser.add_argument("--cnn_lr", type=float, default=1e-3)
    parser.add_argument("--cnn_no_pretrained", action="store_true")
    parser.add_argument("--force_recompute_abcde", action="store_true")
    parser.add_argument("--force_recompute_embeddings", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    for subdir in ["embeddings", "features", "models", "reports", "splits", "predictions", "figures"]:
        ensure_dir(out_dir / subdir)

    target = get_target_spec("cancer")
    df = prepare_task_dataframe(load_ham10000(args.data_root), "cancer")
    df["image_id"] = df["image_id"].astype(str)
    train_df, val_df, test_df = make_lesion_splits(
        df, out_dir, label_col=target.label_column, seed=args.seed
    )
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
    fs = feature_sets["abcd_embeddings"]
    y_train = train_df[target.label_column].to_numpy()
    y_val = val_df[target.label_column].to_numpy()
    y_test = test_df[target.label_column].to_numpy()

    specs = {
        "logistic_regression": (
            LogisticRegression(class_weight="balanced", max_iter=5000, random_state=args.seed),
            {"C": [0.1, 1.0, 10.0]},
        ),
        "knn": (
            KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7, 11]},
        ),
        "neural_net_mlp": (
            MLPClassifier(
                early_stopping=True,
                validation_fraction=0.15,
                max_iter=300,
                random_state=args.seed,
                batch_size=128,
            ),
            {
                "hidden_layer_sizes": [(128,), (256, 64)],
                "alpha": [1e-4, 1e-3],
                "learning_rate_init": [1e-3],
            },
        ),
    }

    metrics: dict[str, dict] = {}
    rows = []
    for model_name, (estimator, grid) in specs.items():
        print(f"[focused] training {model_name} on abcd_embeddings {fs.X_train.shape}", flush=True)
        model, params, val_score = _select_by_validation(estimator, grid, fs.X_train, y_train, fs.X_val, y_val)
        test_metrics, y_pred, y_prob = evaluate_model(model, fs.X_test, y_test, task="cancer")
        metrics[model_name] = {
            "feature_set": "abcd_embeddings",
            "best_params": params,
            "validation_selection_score": val_score,
            "test": test_metrics,
        }
        joblib.dump(model, out_dir / "models" / f"focused_abcd_embeddings_{model_name}.joblib")
        _save_predictions(
            test_df,
            y_test,
            y_pred,
            y_prob,
            out_dir / "predictions" / f"focused_abcd_embeddings_{model_name}_predictions.csv",
        )
        rows.append(_leaderboard_row("abcd_embeddings", model_name, test_metrics))
        print(f"[focused] finished {model_name}: test ROC-AUC={test_metrics.get('ROC-AUC'):.4f}", flush=True)

    if args.run_cnn:
        cnn_metrics = train_cnn_baseline(
            train_df,
            val_df,
            test_df,
            out_dir=out_dir,
            batch_size=args.batch_size,
            epochs=args.cnn_epochs,
            lr=args.cnn_lr,
            seed=args.seed,
            pretrained=not args.cnn_no_pretrained,
            task="cancer",
            device=args.device,
        )
        metrics["cnn_resnet18"] = {"feature_set": "pixels_224", **cnn_metrics}
        rows.append(_leaderboard_row("pixels_224", "cnn_resnet18", cnn_metrics["test"]))

    leaderboard = pd.DataFrame(rows).sort_values(
        by=["ROC-AUC", "balanced_accuracy"], ascending=False, na_position="last"
    )
    leaderboard.to_csv(out_dir / "reports" / "focused_model_leaderboard.csv", index=False)
    write_json(metrics, out_dir / "reports" / "focused_model_metrics.json")
    print(leaderboard.to_string(index=False))
    print(f"Done. Focused leaderboard: {out_dir / 'reports' / 'focused_model_leaderboard.csv'}")


def _select_by_validation(estimator, param_grid, X_train, y_train, X_val, y_val):
    best_model = None
    best_params = {}
    best_score = -np.inf
    grids = list(ParameterGrid(param_grid or {})) or [{}]
    for idx, params in enumerate(grids, start=1):
        print(f"[focused]   params {idx}/{len(grids)}: {params}", flush=True)
        model = clone(estimator).set_params(**params)
        model.fit(X_train, y_train)
        val_metrics, _, _ = evaluate_model(model, X_val, y_val, task="cancer")
        score = val_metrics.get("ROC-AUC", np.nan)
        if not np.isfinite(score):
            score = val_metrics.get("balanced_accuracy", -np.inf)
        if score > best_score:
            best_score = float(score)
            best_model = model
            best_params = dict(params)
    if best_model is None:
        raise RuntimeError(f"Every parameter setting failed for {estimator.__class__.__name__}.")
    return best_model, best_params, best_score


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
    loaded = load_embedding_cache(path)
    if loaded is None:
        raise RuntimeError(f"Failed to load embedding cache after saving: {path}")
    return loaded


def _save_predictions(df, y_true, y_pred, y_prob, path: Path) -> None:
    if y_prob is None:
        probability = np.full(len(y_pred), np.nan)
    elif np.asarray(y_prob).ndim == 1:
        probability = y_prob
    else:
        probability = np.max(y_prob, axis=1)
    pd.DataFrame(
        {
            "image_id": df["image_id"].to_numpy(),
            "true_label": y_true,
            "predicted_label": y_pred,
            "probability": probability,
        }
    ).to_csv(path, index=False)


def _leaderboard_row(feature_set: str, model_name: str, metrics: dict) -> dict:
    return {
        "feature_set": feature_set,
        "model": model_name,
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "sensitivity": metrics.get("sensitivity"),
        "specificity": metrics.get("specificity"),
        "precision": metrics.get("precision"),
        "F1": metrics.get("F1"),
        "ROC-AUC": metrics.get("ROC-AUC"),
        "Average Precision": metrics.get("Average Precision"),
    }


if __name__ == "__main__":
    main()
