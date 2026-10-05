"""Training loop for cached tabular feature sets."""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.tree import export_text

from src.data.targets import get_target_spec
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
    task: str = "cancer",
    seed: int = 42,
    model_names: list[str] | None = None,
    imbalance_strategy: str = "hybrid",
) -> tuple[dict, pd.DataFrame]:
    """Train all registry models on each feature set and save paper-ready artifacts."""
    out_dir = Path(out_dir)
    ensure_dir(out_dir / "models")
    ensure_dir(out_dir / "reports")
    ensure_dir(out_dir / "predictions")
    ensure_dir(out_dir / "figures")

    target = get_target_spec(task)
    label_col = target.label_column
    y_train = train_df[label_col].to_numpy()
    y_val = val_df[label_col].to_numpy()
    y_test = test_df[label_col].to_numpy()

    registry = get_model_registry(seed, use_class_weights=imbalance_strategy in {"hybrid", "class_weight"})
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
            started = time.perf_counter()
            best_model, best_params, val_score = _select_by_validation(
                spec.estimator,
                spec.param_grid,
                feature_set.X_train,
                y_train,
                feature_set.X_val,
                y_val,
                task=task,
                imbalance_strategy=imbalance_strategy,
                seed=seed,
            )
            model_path = out_dir / "models" / f"{feature_name}_{model_name}.joblib"
            joblib.dump(best_model, model_path)

            val_metrics, _, _ = evaluate_model(best_model, feature_set.X_val, y_val, task=task)
            test_metrics, y_pred, y_prob = evaluate_model(best_model, feature_set.X_test, y_test, task=task)
            training_seconds = time.perf_counter() - started
            metrics[feature_name][model_name] = {
                "best_params": best_params,
                "target": target.to_dict(),
                "imbalance_strategy": _effective_imbalance_strategy(model_name, imbalance_strategy),
                "training_seconds": training_seconds,
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
                target,
            )
            _save_eval_figures(feature_name, model_name, y_test, y_pred, y_prob, out_dir)
            leaderboard_rows.append(
                _leaderboard_row(feature_name, model_name, test_metrics, training_seconds=training_seconds)
            )

            if model_name == "decision_tree":
                _save_tree_artifacts(best_model, feature_set, feature_name, out_dir, target.class_names)
            if model_name == "logistic_regression":
                coefficient_rows.extend(_coefficient_rows(best_model, feature_set, feature_name, model_name))
            if hasattr(best_model, "feature_importances_"):
                importance_rows.extend(_importance_rows(best_model, feature_set, feature_name, model_name))
            if model_name == "knn":
                _save_knn_examples(
                    best_model, feature_set, test_df, train_df, y_test, out_dir, feature_name, label_col
                )
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


def _select_by_validation(
    estimator,
    param_grid,
    X_train,
    y_train,
    X_val,
    y_val,
    task: str,
    imbalance_strategy: str = "hybrid",
    seed: int = 42,
):
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
        if (
            model.__class__.__name__ == "XGBClassifier"
            and get_target_spec(task).is_binary
            and imbalance_strategy in {"hybrid", "class_weight"}
        ):
            positives = max(float(np.sum(y_train == 1)), 1.0)
            negatives = max(float(np.sum(y_train == 0)), 1.0)
            model.set_params(scale_pos_weight=negatives / positives)
        fit_X, fit_y = _training_data_for_imbalance(
            model, X_train, y_train, imbalance_strategy=imbalance_strategy, seed=seed
        )
        try:
            model.fit(fit_X, fit_y)
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


def _save_predictions(df, y_true, y_pred, y_prob, path: Path, target) -> None:
    if y_prob is None:
        probability = np.full(len(y_pred), np.nan)
    elif np.asarray(y_prob).ndim == 1:
        probability = y_prob
    else:
        probability = np.max(y_prob, axis=1)
    data = {
        "image_id": df["image_id"].to_numpy(),
        "lesion_id": df["lesion_id"].to_numpy(),
        "dx": df["dx"].to_numpy(),
        "target": target.name,
        "true_label": y_true,
        "predicted_label": y_pred,
        "probability": probability,
    }
    if target.is_binary:
        data[f"probability_{target.class_names[1]}"] = probability
    pd.DataFrame(data).to_csv(path, index=False)


def _leaderboard_row(feature_set: str, model_name: str, metrics: dict, training_seconds: float) -> dict:
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
        "AUPRC": metrics.get("Average Precision"),
        "training_seconds": training_seconds,
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


def _save_tree_artifacts(
    model, feature_set: FeatureSet, feature_name: str, out_dir: Path, class_names: tuple[str, ...]
) -> None:
    tree_text = export_text(model, feature_names=feature_set.feature_names, max_depth=15)
    path = out_dir / "reports" / f"{feature_name}_decision_tree_rules.txt"
    path.write_text(tree_text, encoding="utf-8")
    save_tree_plot(
        model,
        feature_set.feature_names,
        out_dir / "figures" / f"{feature_name}_decision_tree.png",
        class_names=class_names,
    )
    if feature_name == "abcd_embeddings":
        (out_dir / "reports" / "tree_rules.txt").write_text(tree_text, encoding="utf-8")
        save_tree_plot(
            model, feature_set.feature_names, out_dir / "figures" / "decision_tree.png", class_names=class_names
        )


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


def _save_knn_examples(
    model, feature_set: FeatureSet, test_df, train_df, y_test, out_dir: Path, feature_name: str, label_col: str
):
    try:
        distances, indices = model.kneighbors(feature_set.X_test[: min(10, len(test_df))], n_neighbors=5)
    except Exception:
        return
    rows = []
    for test_i, (dists, neighbor_indices) in enumerate(zip(distances, indices)):
        for rank, (dist, train_i) in enumerate(zip(dists, neighbor_indices), start=1):
            # Random oversampling (used for KNN under ``hybrid``) can make the
            # fitted matrix longer than the original training dataframe.  In
            # that case there is no unique source image to report; retain the
            # neighbor label from the fitted estimator and use an explicit
            # synthetic row identifier instead of indexing past train_df.
            train_i = int(train_i)
            if train_i < len(train_df):
                neighbor_image_id = train_df.iloc[train_i]["image_id"]
                neighbor_label = train_df.iloc[train_i][label_col]
            else:
                neighbor_image_id = f"oversampled_train_row_{train_i}"
                fitted_labels = getattr(model, "_y", None)
                neighbor_label = fitted_labels[train_i] if fitted_labels is not None else np.nan
            rows.append(
                {
                    "test_image_id": test_df.iloc[test_i]["image_id"],
                    "test_label": y_test[test_i],
                    "neighbor_rank": rank,
                    "neighbor_image_id": neighbor_image_id,
                    "neighbor_label": neighbor_label,
                    "distance": float(dist),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "reports" / f"{feature_name}_knn_neighbors.csv", index=False)


def _effective_imbalance_strategy(model_name: str, requested: str) -> str:
    if requested == "hybrid":
        return "random_oversampling" if model_name == "knn" else "class_weight"
    return requested


def _training_data_for_imbalance(model, X, y, imbalance_strategy: str, seed: int):
    if imbalance_strategy not in {"none", "class_weight", "oversample", "hybrid"}:
        raise ValueError(f"Unknown imbalance strategy: {imbalance_strategy}")
    should_oversample = imbalance_strategy == "oversample" or (
        imbalance_strategy == "hybrid" and model.__class__.__name__ == "KNeighborsClassifier"
    )
    if not should_oversample:
        return X, y
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    target_count = int(counts.max())
    indices = []
    for class_value, count in zip(classes, counts, strict=True):
        class_indices = np.flatnonzero(y == class_value)
        indices.extend(class_indices.tolist())
        if count < target_count:
            indices.extend(rng.choice(class_indices, target_count - count, replace=True).tolist())
    indices = np.asarray(indices)
    rng.shuffle(indices)
    return np.asarray(X)[indices], y[indices]
