#!/usr/bin/env python
"""Train only decision trees across feature sets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.tree import DecisionTreeClassifier, export_text

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
from src.utils.io import ensure_dir, write_json
from src.utils.plotting import save_tree_plot
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train decision trees across all cached feature sets.")
    parser.add_argument("--data_root", default="data/HAM10000")
    parser.add_argument("--embedding_model_name", default="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--pca_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_depths", nargs="+", type=int, default=[2, 3, 4, 5, 10, 15])
    parser.add_argument("--min_samples_leaf", nargs="+", type=int, default=[5, 10, 20])
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

    y_train = train_df[target.label_column].to_numpy()
    y_val = val_df[target.label_column].to_numpy()
    y_test = test_df[target.label_column].to_numpy()
    grid = {
        "max_depth": args.max_depths,
        "min_samples_leaf": args.min_samples_leaf,
    }
    rows = []
    metrics = {}
    for feature_name, fs in feature_sets.items():
        print(f"[trees] training {feature_name} decision tree on {fs.X_train.shape}", flush=True)
        model, params, val_score = _select_tree(fs.X_train, y_train, fs.X_val, y_val, grid, args.seed)
        test_metrics, y_pred, y_prob = evaluate_model(model, fs.X_test, y_test, task="cancer")
        metrics[feature_name] = {
            "best_params": params,
            "validation_selection_score": val_score,
            "test": test_metrics,
        }
        joblib.dump(model, out_dir / "models" / f"{feature_name}_decision_tree_depth_search.joblib")
        _save_predictions(
            test_df,
            y_test,
            y_pred,
            y_prob,
            out_dir / "predictions" / f"{feature_name}_decision_tree_depth_search_predictions.csv",
        )
        rules = export_text(model, feature_names=fs.feature_names, max_depth=15)
        (out_dir / "reports" / f"{feature_name}_decision_tree_depth_search_rules.txt").write_text(
            rules, encoding="utf-8"
        )
        save_tree_plot(model, fs.feature_names, out_dir / "figures" / f"{feature_name}_decision_tree_depth_search.png")
        if feature_name == "abcd_embeddings":
            (out_dir / "reports" / "tree_rules.txt").write_text(rules, encoding="utf-8")
            save_tree_plot(model, fs.feature_names, out_dir / "figures" / "decision_tree.png")
        row = {
            "feature_set": feature_name,
            "model": "decision_tree",
            "best_max_depth": params["max_depth"],
            "best_min_samples_leaf": params["min_samples_leaf"],
            "validation_ROC-AUC": val_score,
            "accuracy": test_metrics.get("accuracy"),
            "balanced_accuracy": test_metrics.get("balanced_accuracy"),
            "sensitivity": test_metrics.get("sensitivity"),
            "specificity": test_metrics.get("specificity"),
            "precision": test_metrics.get("precision"),
            "F1": test_metrics.get("F1"),
            "ROC-AUC": test_metrics.get("ROC-AUC"),
            "Average Precision": test_metrics.get("Average Precision"),
        }
        rows.append(row)
        print(
            f"[trees] finished {feature_name}: params={params}, test ROC-AUC={test_metrics.get('ROC-AUC'):.4f}",
            flush=True,
        )

    leaderboard = pd.DataFrame(rows).sort_values(
        by=["ROC-AUC", "balanced_accuracy"], ascending=False, na_position="last"
    )
    leaderboard.to_csv(out_dir / "reports" / "decision_tree_depth_leaderboard.csv", index=False)
    write_json(metrics, out_dir / "reports" / "decision_tree_depth_metrics.json")
    print(leaderboard.to_string(index=False))
    print(f"Done. Tree leaderboard: {out_dir / 'reports' / 'decision_tree_depth_leaderboard.csv'}")


def _select_tree(X_train, y_train, X_val, y_val, grid, seed: int):
    best_model = None
    best_params = {}
    best_score = -np.inf
    params_list = list(ParameterGrid(grid))
    for idx, params in enumerate(params_list, start=1):
        print(f"[trees]   params {idx}/{len(params_list)}: {params}", flush=True)
        model = DecisionTreeClassifier(class_weight="balanced", random_state=seed, **params)
        model.fit(X_train, y_train)
        val_metrics, _, _ = evaluate_model(model, X_val, y_val, task="cancer")
        score = val_metrics.get("ROC-AUC", np.nan)
        if not np.isfinite(score):
            score = val_metrics.get("balanced_accuracy", -np.inf)
        if score > best_score:
            best_model = model
            best_params = dict(params)
            best_score = float(score)
    if best_model is None:
        raise RuntimeError("Every decision-tree parameter setting failed.")
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


if __name__ == "__main__":
    main()
