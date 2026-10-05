#!/usr/bin/env python
"""Readable decision-tree sweep with PCA dimension and threshold tuning.

This keeps the older tree experiments intact and writes separate artifacts.

- outputs/reports/{run_name}_candidates.csv
- outputs/reports/{run_name}_leaderboard.csv
- outputs/reports/{run_name}_metrics.json
- outputs/reports/{run_name}_best_rules.txt
- outputs/figures/{run_name}_best.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
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
from src.utils.metrics import binary_metrics
from src.utils.plotting import save_tree_plot
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep readable ABCD + PCA(embeddings) decision trees.")
    parser.add_argument("--data_root", default="data/HAM10000")
    parser.add_argument("--embedding_model_name", default="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run_name", default="readable_tree_tuned")
    parser.add_argument("--pca_dims", nargs="+", type=int, default=[4, 8, 16, 32, 64, 128])
    parser.add_argument("--max_depths", nargs="+", type=int, default=[3, 4, 5])
    parser.add_argument("--min_samples_leaf", nargs="+", type=int, default=[5, 10, 20, 50, 100, 200])
    parser.add_argument("--min_samples_split", nargs="+", type=int, default=[2, 20, 50])
    parser.add_argument("--criteria", nargs="+", default=["gini", "entropy", "log_loss"])
    parser.add_argument("--max_features", nargs="+", default=["none", "sqrt", "log2"])
    parser.add_argument("--class_weights", nargs="+", default=["balanced", "none"])
    parser.add_argument("--ccp_alphas", nargs="+", type=float, default=[0.0, 0.0005, 0.001, 0.002])
    parser.add_argument(
        "--threshold_objective",
        choices=["balanced_accuracy", "f1", "sensitivity_at_specificity"],
        default="balanced_accuracy",
    )
    parser.add_argument("--min_specificity", type=float, default=0.70)
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

    y_train = train_df[target.label_column].to_numpy()
    y_val = val_df[target.label_column].to_numpy()
    y_test = test_df[target.label_column].to_numpy()
    run_name = _safe_run_name(args.run_name)
    grid = list(
        ParameterGrid(
            {
                "criterion": args.criteria,
                "max_depth": args.max_depths,
                "min_samples_leaf": args.min_samples_leaf,
                "min_samples_split": args.min_samples_split,
                "max_features": [_parse_optional_string(v) for v in args.max_features],
                "class_weight": [_parse_class_weight(v) for v in args.class_weights],
                "ccp_alpha": args.ccp_alphas,
            }
        )
    )
    print(
        f"[readable_tree] run_name={run_name}, pca_dims={args.pca_dims}, "
        f"candidate_count={len(args.pca_dims) * len(grid)}",
        flush=True,
    )

    candidate_rows = []
    best = None
    for pca_dim in args.pca_dims:
        print(f"[readable_tree] building feature set with pca_dim={pca_dim}", flush=True)
        feature_sets = build_feature_sets(
            train_df,
            val_df,
            test_df,
            abcde_df,
            embedding_cache,
            out_dir=out_dir,
            pca_dim=pca_dim,
        )
        fs = feature_sets["abcd_pca_embeddings"]
        for idx, params in enumerate(grid, start=1):
            print(f"[readable_tree] pca_dim={pca_dim} params {idx}/{len(grid)}: {params}", flush=True)
            model = DecisionTreeClassifier(random_state=args.seed, **params)
            model.fit(fs.X_train, y_train)
            val_pred, val_prob = _predict_with_prob(model, fs.X_val)
            val_metrics = binary_metrics(y_val, val_pred, val_prob, class_names=target.class_names)
            threshold, threshold_metrics = _select_threshold(
                y_val,
                val_prob,
                objective=args.threshold_objective,
                min_specificity=args.min_specificity,
            )
            test_pred, test_prob = _predict_with_prob(model, fs.X_test, threshold=threshold)
            test_metrics = binary_metrics(y_test, test_pred, test_prob, class_names=target.class_names)

            row = {
                "feature_set": "abcd_pca_embeddings",
                "model": "readable_decision_tree",
                "pca_dim": pca_dim,
                "criterion": params["criterion"],
                "max_depth": params["max_depth"],
                "min_samples_leaf": params["min_samples_leaf"],
                "min_samples_split": params["min_samples_split"],
                "max_features": _format_optional(params["max_features"]),
                "class_weight": _format_optional(params["class_weight"]),
                "ccp_alpha": params["ccp_alpha"],
                "threshold": threshold,
                "threshold_objective": args.threshold_objective,
                "validation_ROC-AUC": val_metrics.get("ROC-AUC"),
                "validation_balanced_accuracy_at_0.5": val_metrics.get("balanced_accuracy"),
                "validation_threshold_score": threshold_metrics["selection_score"],
                "validation_balanced_accuracy": threshold_metrics["balanced_accuracy"],
                "validation_sensitivity": threshold_metrics["sensitivity"],
                "validation_specificity": threshold_metrics["specificity"],
                "accuracy": test_metrics.get("accuracy"),
                "balanced_accuracy": test_metrics.get("balanced_accuracy"),
                "sensitivity": test_metrics.get("sensitivity"),
                "specificity": test_metrics.get("specificity"),
                "precision": test_metrics.get("precision"),
                "F1": test_metrics.get("F1"),
                "ROC-AUC": test_metrics.get("ROC-AUC"),
                "Average Precision": test_metrics.get("Average Precision"),
            }
            candidate_rows.append(row)
            selection_score = (
                row["validation_ROC-AUC"],
                row["validation_threshold_score"],
                -row["max_depth"],
                -row["min_samples_leaf"],
            )
            if best is None or selection_score > best["selection_score"]:
                best = {
                    "selection_score": selection_score,
                    "model": model,
                    "feature_set": fs,
                    "pca_dim": pca_dim,
                    "params": dict(params),
                    "threshold": threshold,
                    "val_metrics": val_metrics,
                    "threshold_metrics": threshold_metrics,
                    "test_metrics": test_metrics,
                    "test_pred": test_pred,
                    "test_prob": test_prob,
                }

    if best is None:
        raise RuntimeError("No readable tree candidates were trained.")

    candidates = pd.DataFrame(candidate_rows).sort_values(
        by=["validation_ROC-AUC", "validation_threshold_score", "ROC-AUC"],
        ascending=False,
        na_position="last",
    )
    candidates.to_csv(out_dir / "reports" / f"{run_name}_candidates.csv", index=False)

    leaderboard = pd.DataFrame(
        [
            {
                "feature_set": "abcd_pca_embeddings",
                "model": "readable_decision_tree",
                "pca_dim": best["pca_dim"],
                "best_criterion": best["params"]["criterion"],
                "best_max_depth": best["params"]["max_depth"],
                "best_min_samples_leaf": best["params"]["min_samples_leaf"],
                "best_min_samples_split": best["params"]["min_samples_split"],
                "best_max_features": _format_optional(best["params"]["max_features"]),
                "best_class_weight": _format_optional(best["params"]["class_weight"]),
                "best_ccp_alpha": best["params"]["ccp_alpha"],
                "threshold": best["threshold"],
                "validation_ROC-AUC": best["val_metrics"].get("ROC-AUC"),
                "validation_balanced_accuracy": best["threshold_metrics"]["balanced_accuracy"],
                "validation_sensitivity": best["threshold_metrics"]["sensitivity"],
                "validation_specificity": best["threshold_metrics"]["specificity"],
                "accuracy": best["test_metrics"].get("accuracy"),
                "balanced_accuracy": best["test_metrics"].get("balanced_accuracy"),
                "sensitivity": best["test_metrics"].get("sensitivity"),
                "specificity": best["test_metrics"].get("specificity"),
                "precision": best["test_metrics"].get("precision"),
                "F1": best["test_metrics"].get("F1"),
                "ROC-AUC": best["test_metrics"].get("ROC-AUC"),
                "Average Precision": best["test_metrics"].get("Average Precision"),
            }
        ]
    )
    leaderboard.to_csv(out_dir / "reports" / f"{run_name}_leaderboard.csv", index=False)

    joblib.dump(best["model"], out_dir / "models" / f"{run_name}_best.joblib")
    rules = export_text(best["model"], feature_names=best["feature_set"].feature_names, max_depth=15)
    (out_dir / "reports" / f"{run_name}_best_rules.txt").write_text(rules, encoding="utf-8")
    save_tree_plot(best["model"], best["feature_set"].feature_names, out_dir / "figures" / f"{run_name}_best.png")
    _save_predictions(
        test_df,
        y_test,
        best["test_pred"],
        best["test_prob"],
        out_dir / "predictions" / f"{run_name}_best_predictions.csv",
    )
    write_json(
        {
            "best": {
                "feature_set": "abcd_pca_embeddings",
                "pca_dim": best["pca_dim"],
                "params": best["params"],
                "threshold": best["threshold"],
                "threshold_objective": args.threshold_objective,
                "validation": best["threshold_metrics"],
                "test": best["test_metrics"],
            }
        },
        out_dir / "reports" / f"{run_name}_metrics.json",
    )
    print(leaderboard.to_string(index=False))
    print(f"Done. Readable-tree leaderboard: {out_dir / 'reports' / f'{run_name}_leaderboard.csv'}")


def _predict_with_prob(model, X, threshold: float = 0.5):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    return pred, prob


def _select_threshold(y_true, y_prob, objective: str, min_specificity: float) -> tuple[float, dict]:
    thresholds = np.unique(np.r_[np.linspace(0.01, 0.99, 99), y_prob])
    best_threshold = 0.5
    best_metrics = None
    best_score = -np.inf
    for threshold in thresholds:
        pred = (y_prob >= threshold).astype(int)
        cm = _confusion(y_true, pred)
        tn, fp, fn, tp = cm
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        balanced_accuracy = balanced_accuracy_score(y_true, pred)
        if objective == "balanced_accuracy":
            score = balanced_accuracy
        elif objective == "f1":
            score = f1_score(y_true, pred, zero_division=0)
        else:
            score = sensitivity if specificity >= min_specificity else -np.inf
        metrics = {
            "threshold": float(threshold),
            "selection_score": float(score),
            "balanced_accuracy": float(balanced_accuracy),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "F1": float(f1_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        }
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
            best_metrics = metrics
    if best_metrics is None:
        pred = (y_prob >= 0.5).astype(int)
        best_metrics = {
            "threshold": 0.5,
            "selection_score": float(balanced_accuracy_score(y_true, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            "sensitivity": float(recall_score(y_true, pred, zero_division=0)),
            "specificity": 0.0,
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "F1": float(f1_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "confusion_matrix": _confusion_matrix_list(y_true, pred),
        }
    return best_threshold, best_metrics


def _confusion(y_true, y_pred) -> tuple[int, int, int, int]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return tn, fp, fn, tp


def _confusion_matrix_list(y_true, y_pred) -> list[list[int]]:
    tn, fp, fn, tp = _confusion(y_true, y_pred)
    return [[tn, fp], [fn, tp]]


def _parse_optional_string(value: str):
    if value.lower() in {"none", "null"}:
        return None
    return value


def _parse_class_weight(value: str):
    if value.lower() in {"none", "null"}:
        return None
    if value == "balanced":
        return "balanced"
    raise ValueError(f"Unsupported class_weight option: {value}")


def _format_optional(value) -> str:
    if value is None:
        return "none"
    return str(value)


def _safe_run_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip())
    return safe or "readable_tree_tuned"


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
    pd.DataFrame(
        {
            "image_id": df["image_id"].to_numpy(),
            "true_label": y_true,
            "predicted_label": y_pred,
            "probability": y_prob,
        }
    ).to_csv(path, index=False)


if __name__ == "__main__":
    main()
