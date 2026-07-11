"""Training loop for cached tabular feature sets."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.tree import export_text

from src.features.fusion import FeatureSet
from src.models.registry import get_model_registry
from src.training.evaluate import evaluate_model, predict_labels_and_probabilities
from src.utils.io import ensure_dir, write_json
from src.utils.plotting import save_confusion_matrix, save_roc_curve, save_tree_plot


def train_all_tabular(
    feature_sets: dict[str, FeatureSet],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: str | Path,
    task: str = "binary",
    seed: int = 42,
    model_names: list[str] | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Train all registry models on each feature set and save paper-ready artifacts."""
    out_dir = Path(out_dir)
    ensure_dir(out_dir / "models")
    ensure_dir(out_dir / "reports")
    ensure_dir(out_dir / "predictions")
    ensure_dir(out_dir / "figures")

    label_col = "binary_label" if task == "binary" else "multiclass_label"
    y_train = train_df[label_col].to_numpy()
    y_val = val_df[label_col].to_numpy()
    y_test = test_df[label_col].to_numpy()

    registry = get_model_registry(seed)
    if model_names is not None:
        requested = set(model_names)
        registry = {name: spec for name, spec in registry.items() if name in requested}
        missing = sorted(requested - set(registry))
        if missing:
            raise KeyError(f"Requested tabular models are not available: {missing}")
    metrics: dict[str, dict] = {}
    leaderboard_rows = []
    coefficient_rows = []
    importance_rows = []

    for feature_name, feature_set in feature_sets.items():
        metrics[feature_name] = {}
        print(
            f"[train_tabular] Feature set {feature_name}: "
            f"train={feature_set.X_train.shape}, val={feature_set.X_val.shape}, test={feature_set.X_test.shape}",
            flush=True,
        )
        for model_name, spec in registry.items():
            print(f"[train_tabular] Fitting {feature_name}/{model_name}", flush=True)
            best_model, best_params, val_score = _select_by_validation(
                spec.estimator,
                spec.param_grid,
                feature_set.X_train,
                y_train,
                feature_set.X_val,
                y_val,
                task=task,
            )
            model_path = out_dir / "models" / f"{feature_name}_{model_name}.joblib"
            joblib.dump(best_model, model_path)

            val_metrics, _, _ = evaluate_model(best_model, feature_set.X_val, y_val, task=task)
            test_metrics, y_pred, y_prob = evaluate_model(best_model, feature_set.X_test, y_test, task=task)
            metrics[feature_name][model_name] = {
                "best_params": best_params,
                "validation_selection_score": val_score,
                "validation": val_metrics,
                "test": test_metrics,
            }

            _save_predictions(
                test_df,
                y_test,
                y_pred,
                y_prob,
                out_dir / "predictions" / f"{feature_name}_{model_name}_predictions.csv",
            )
            _save_eval_figures(feature_name, model_name, y_test, y_pred, y_prob, out_dir)
            leaderboard_rows.append(_leaderboard_row(feature_name, model_name, test_metrics))

            if model_name == "decision_tree":
                _save_tree_artifacts(best_model, feature_set, feature_name, out_dir)
            if model_name == "logistic_regression":
                coefficient_rows.extend(_coefficient_rows(best_model, feature_set, feature_name, model_name))
            if hasattr(best_model, "feature_importances_"):
                importance_rows.extend(_importance_rows(best_model, feature_set, feature_name, model_name))
            if model_name == "knn":
                _save_knn_examples(best_model, feature_set, test_df, train_df, y_test, out_dir, feature_name)
            print(
                f"[train_tabular] Finished {feature_name}/{model_name}: "
                f"best_params={best_params}, val_score={val_score:.4f}, "
                f"test_auc={test_metrics.get('ROC-AUC')}",
                flush=True,
            )

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        by=["ROC-AUC", "balanced_accuracy"], ascending=False, na_position="last"
    )
    leaderboard.to_csv(out_dir / "reports" / "leaderboard.csv", index=False)
    write_json(metrics, out_dir / "reports" / "metrics.json")

    if coefficient_rows:
        pd.DataFrame(coefficient_rows).sort_values("abs_coefficient", ascending=False).to_csv(
            out_dir / "reports" / "logistic_coefficients.csv", index=False
        )
    if importance_rows:
        pd.DataFrame(importance_rows).sort_values("importance", ascending=False).to_csv(
            out_dir / "reports" / "feature_importances.csv", index=False
        )
    return metrics, leaderboard


def _select_by_validation(estimator, param_grid, X_train, y_train, X_val, y_val, task: str):
    best_model = None
    best_params = {}
    best_score = -np.inf
    grids = list(ParameterGrid(param_grid or {})) or [{}]
    for idx, params in enumerate(grids, start=1):
        print(
            f"[train_tabular]   {estimator.__class__.__name__} params {idx}/{len(grids)}: {params}",
            flush=True,
        )
        model = clone(estimator).set_params(**params)
        if model.__class__.__name__ == "XGBClassifier" and task == "binary":
            positives = max(float(np.sum(y_train == 1)), 1.0)
            negatives = max(float(np.sum(y_train == 0)), 1.0)
            model.set_params(scale_pos_weight=negatives / positives)
        try:
            model.fit(X_train, y_train)
            val_metrics, _, _ = evaluate_model(model, X_val, y_val, task=task)
            score = val_metrics.get("ROC-AUC", np.nan)
            if not np.isfinite(score):
                score = val_metrics.get("balanced_accuracy", -np.inf)
        except Exception:
            continue
        if score > best_score:
            best_score = float(score)
            best_model = model
            best_params = dict(params)
    if best_model is None:
        raise RuntimeError(f"Every parameter setting failed for {estimator.__class__.__name__}.")
    return best_model, best_params, best_score


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


def _save_eval_figures(feature_set, model_name, y_true, y_pred, y_prob, out_dir: Path) -> None:
    try:
        save_confusion_matrix(y_true, y_pred, out_dir / "figures" / f"{feature_set}_{model_name}_confusion.png")
    except Exception:
        pass
    if y_prob is not None and np.asarray(y_prob).ndim == 1:
        try:
            save_roc_curve(y_true, y_prob, out_dir / "figures" / f"{feature_set}_{model_name}_roc.png")
        except Exception:
            pass


def _save_tree_artifacts(model, feature_set: FeatureSet, feature_name: str, out_dir: Path) -> None:
    tree_text = export_text(model, feature_names=feature_set.feature_names, max_depth=15)
    path = out_dir / "reports" / f"{feature_name}_decision_tree_rules.txt"
    path.write_text(tree_text, encoding="utf-8")
    save_tree_plot(model, feature_set.feature_names, out_dir / "figures" / f"{feature_name}_decision_tree.png")
    if feature_name == "abcde_embeddings":
        (out_dir / "reports" / "tree_rules.txt").write_text(tree_text, encoding="utf-8")
        save_tree_plot(model, feature_set.feature_names, out_dir / "figures" / "decision_tree.png")


def _coefficient_rows(model, feature_set: FeatureSet, feature_name: str, model_name: str) -> list[dict]:
    coefs = np.ravel(model.coef_)
    return [
        {
            "feature_set": feature_name,
            "model": model_name,
            "feature": feature,
            "coefficient": float(coef),
            "abs_coefficient": float(abs(coef)),
        }
        for feature, coef in zip(feature_set.feature_names, coefs)
    ]


def _importance_rows(model, feature_set: FeatureSet, feature_name: str, model_name: str) -> list[dict]:
    return [
        {
            "feature_set": feature_name,
            "model": model_name,
            "feature": feature,
            "importance": float(importance),
        }
        for feature, importance in zip(feature_set.feature_names, model.feature_importances_)
    ]


def _save_knn_examples(model, feature_set: FeatureSet, test_df, train_df, y_test, out_dir: Path, feature_name: str):
    try:
        distances, indices = model.kneighbors(feature_set.X_test[: min(10, len(test_df))], n_neighbors=5)
    except Exception:
        return
    rows = []
    for test_i, (dists, neighbor_indices) in enumerate(zip(distances, indices)):
        for rank, (dist, train_i) in enumerate(zip(dists, neighbor_indices), start=1):
            rows.append(
                {
                    "test_image_id": test_df.iloc[test_i]["image_id"],
                    "test_label": y_test[test_i],
                    "neighbor_rank": rank,
                    "neighbor_image_id": train_df.iloc[int(train_i)]["image_id"],
                    "neighbor_label": train_df.iloc[int(train_i)]["binary_label"],
                    "distance": float(dist),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "reports" / f"{feature_name}_knn_neighbors.csv", index=False)
