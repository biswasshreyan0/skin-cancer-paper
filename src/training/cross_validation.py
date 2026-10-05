"""Repeated, lesion-grouped nested cross-validation for tabular models."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from src.data.targets import get_target_spec
from src.features.fusion import build_feature_sets
from src.models.registry import get_model_registry
from src.training.evaluate import evaluate_model
from src.training.train_tabular import _select_by_validation, _training_data_for_imbalance
from src.utils.io import ensure_dir, write_json


def run_repeated_group_cv(
    development_df: pd.DataFrame,
    abcd_df: pd.DataFrame,
    embedding_cache: dict[str, np.ndarray],
    out_dir: str | Path,
    task: str = "cancer",
    pca_dim: int = 64,
    seed: int = 42,
    folds: int = 5,
    repeats: int = 3,
    model_names: list[str] | None = None,
    imbalance_strategy: str = "hybrid",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run nested CV on development data while leaving the final test split untouched."""
    if folds < 2 or repeats < 1:
        raise ValueError("Cross-validation requires folds >= 2 and repeats >= 1.")
    target = get_target_spec(task)
    label_col = target.label_column
    registry = get_model_registry(seed, use_class_weights=imbalance_strategy in {"hybrid", "class_weight"})
    if model_names is not None:
        requested = set(model_names)
        registry = {name: spec for name, spec in registry.items() if name in requested}
        missing = sorted(requested - set(registry))
        if missing:
            raise KeyError(f"Requested CV models are unavailable: {missing}")

    out_dir = Path(out_dir)
    reports_dir = ensure_dir(out_dir / "reports")
    prediction_root = ensure_dir(out_dir / "predictions" / "cross_validation")
    progress_path = reports_dir / "cv_progress.csv"
    state_path = reports_dir / "cv_progress_state.json"
    progress_state = {
        "target": target.name,
        "pca_dim": pca_dim,
        "seed": seed,
        "folds": folds,
        "repeats": repeats,
        "model_names": sorted(registry),
        "imbalance_strategy": imbalance_strategy,
        "development_images": len(development_df),
        "development_lesions": int(development_df["lesion_id"].nunique()),
    }
    completed: dict[str, dict] = {}
    try:
        prior_state = json.loads(state_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, TypeError, OSError):
        prior_state = None
    if prior_state == progress_state and progress_path.exists():
        try:
            prior_rows = pd.read_csv(progress_path).to_dict(orient="records")
            completed = {str(row["job_key"]): row for row in prior_rows if row.get("job_key")}
            print(f"[cross_validation] Resuming {len(completed)} completed jobs from {progress_path}", flush=True)
        except (OSError, ValueError, KeyError):
            completed = {}
    write_json(progress_state, state_path)
    rows: list[dict] = list(completed.values())
    y_all = development_df[label_col].to_numpy()
    groups = development_df["lesion_id"].to_numpy()

    for repeat in range(repeats):
        outer = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
        for fold, (outer_train_idx, outer_val_idx) in enumerate(
            outer.split(development_df, y_all, groups), start=1
        ):
            outer_train = development_df.iloc[outer_train_idx].copy()
            outer_val = development_df.iloc[outer_val_idx].copy()
            inner_train, inner_val = _inner_group_split(outer_train, label_col, seed + repeat * folds + fold)

            inner_sets = build_feature_sets(
                inner_train,
                inner_val,
                outer_val,
                abcd_df,
                embedding_cache,
                out_dir=out_dir,
                pca_dim=pca_dim,
                save_artifacts=False,
            )
            outer_sets = build_feature_sets(
                outer_train,
                outer_val,
                outer_val,
                abcd_df,
                embedding_cache,
                out_dir=out_dir,
                pca_dim=pca_dim,
                save_artifacts=False,
            )
            fold_dir = ensure_dir(prediction_root / f"repeat_{repeat + 1:02d}_fold_{fold:02d}")
            y_inner_train = inner_train[label_col].to_numpy()
            y_inner_val = inner_val[label_col].to_numpy()
            y_outer_train = outer_train[label_col].to_numpy()
            y_outer_val = outer_val[label_col].to_numpy()

            for feature_name, inner_feature_set in inner_sets.items():
                outer_feature_set = outer_sets[feature_name]
                for model_name, spec in registry.items():
                    job_key = f"repeat_{repeat + 1:02d}_fold_{fold:02d}/{feature_name}/{model_name}"
                    prediction_path = fold_dir / f"{feature_name}_{model_name}.csv"
                    if job_key in completed and prediction_path.exists():
                        print(f"[cross_validation] skipping completed {job_key}", flush=True)
                        continue
                    print(
                        f"[cross_validation] repeat={repeat + 1}/{repeats} fold={fold}/{folds} "
                        f"{feature_name}/{model_name}",
                        flush=True,
                    )
                    started = time.perf_counter()
                    _, best_params, _ = _select_by_validation(
                        spec.estimator,
                        spec.param_grid,
                        inner_feature_set.X_train,
                        y_inner_train,
                        inner_feature_set.X_val,
                        y_inner_val,
                        task=task,
                        imbalance_strategy=imbalance_strategy,
                        seed=seed + repeat * folds + fold,
                    )
                    model = clone(spec.estimator).set_params(**best_params)
                    if (
                        model.__class__.__name__ == "XGBClassifier"
                        and target.is_binary
                        and imbalance_strategy in {"hybrid", "class_weight"}
                    ):
                        positives = max(float(np.sum(y_outer_train == 1)), 1.0)
                        negatives = max(float(np.sum(y_outer_train == 0)), 1.0)
                        model.set_params(scale_pos_weight=negatives / positives)
                    fit_X, fit_y = _training_data_for_imbalance(
                        model,
                        outer_feature_set.X_train,
                        y_outer_train,
                        imbalance_strategy=imbalance_strategy,
                        seed=seed + repeat * folds + fold,
                    )
                    model.fit(fit_X, fit_y)
                    metrics, y_pred, y_prob = evaluate_model(
                        model, outer_feature_set.X_val, y_outer_val, task=task
                    )
                    elapsed = time.perf_counter() - started
                    rows.append(
                        _fold_metric_row(
                            repeat + 1,
                            fold,
                            feature_name,
                            model_name,
                            best_params,
                            metrics,
                            elapsed,
                            target,
                            outer_train,
                            outer_val,
                        )
                    )
                    rows[-1]["job_key"] = job_key
                    _save_fold_predictions(
                        outer_val,
                        y_outer_val,
                        y_pred,
                        y_prob,
                        prediction_path,
                        target,
                        repeat + 1,
                        fold,
                    )
                    # Persist after every completed model so Ctrl-C, sleep,
                    # or power loss loses at most the currently running job.
                    pd.DataFrame(rows).to_csv(progress_path, index=False)

    fold_metrics = pd.DataFrame(rows)
    fold_metrics.to_csv(reports_dir / "cv_fold_metrics.csv", index=False)
    summary = _summarize_cv(fold_metrics)
    summary.to_csv(reports_dir / "cv_summary.csv", index=False)
    _write_ablation_and_significance(fold_metrics, summary, reports_dir)
    write_json(
        {
            "target": target.to_dict(),
            "outer_protocol": f"{repeats} repeats of {folds}-fold StratifiedGroupKFold by lesion_id",
            "inner_protocol": "grouped five-fold split; first fold used for hyperparameter selection",
            "final_holdout_used_in_cv": False,
            "imbalance_strategy": imbalance_strategy,
            "fold_prediction_root": str(prediction_root),
        },
        reports_dir / "cv_protocol.json",
    )
    return fold_metrics, summary


def _inner_group_split(df: pd.DataFrame, label_col: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, df[label_col], df["lesion_id"]))
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def _fold_metric_row(
    repeat,
    fold,
    feature_set,
    model,
    params,
    metrics,
    training_seconds,
    target,
    train_df,
    val_df,
) -> dict:
    row = {
        "repeat": repeat,
        "fold": fold,
        "target": target.name,
        "feature_set": feature_set,
        "model": model,
        "best_params": str(params),
        "train_images": len(train_df),
        "train_lesions": train_df["lesion_id"].nunique(),
        "validation_images": len(val_df),
        "validation_lesions": val_df["lesion_id"].nunique(),
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "F1": metrics.get("F1", metrics.get("macro F1")),
        "ROC-AUC": metrics.get("ROC-AUC", metrics.get("macro ROC-AUC")),
        "AUPRC": metrics.get("Average Precision", metrics.get("macro AUPRC")),
        "training_seconds": training_seconds,
    }
    for class_name, values in metrics.get("per_class", {}).items():
        for metric_name, value in values.items():
            row[f"{class_name}_{metric_name}"] = value
    return row


def _summarize_cv(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    id_columns = ["target", "feature_set", "model"]
    metric_columns = [
        column
        for column in fold_metrics.select_dtypes(include=[np.number]).columns
        if column not in {"repeat", "fold"}
    ]
    summary = fold_metrics.groupby(id_columns, as_index=False)[metric_columns].agg(["mean", "std"])
    summary.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    return summary.reset_index(drop=True)


def _write_ablation_and_significance(
    fold_metrics: pd.DataFrame, summary: pd.DataFrame, reports_dir: Path
) -> None:
    full_feature_set = "abcd_embeddings_metadata"
    candidates = summary[summary["feature_set"].eq(full_feature_set)]
    if candidates.empty:
        full_feature_set = "abcd_embeddings"
        candidates = summary[summary["feature_set"].eq(full_feature_set)]
    if candidates.empty:
        pd.DataFrame().to_csv(reports_dir / "ablation_table.csv", index=False)
        pd.DataFrame().to_csv(reports_dir / "statistical_tests.csv", index=False)
        return
    best_model = candidates.sort_values("AUPRC_mean", ascending=False).iloc[0]["model"]
    ablation = summary[summary["model"].eq(best_model)].copy()
    baseline = float(ablation.loc[ablation["feature_set"].eq(full_feature_set), "AUPRC_mean"].iloc[0])
    ablation["AUPRC_delta_vs_full"] = ablation["AUPRC_mean"] - baseline
    ablation["reference_feature_set"] = full_feature_set
    ablation.sort_values("AUPRC_mean", ascending=False).to_csv(
        reports_dir / "ablation_table.csv", index=False
    )

    try:
        from scipy.stats import wilcoxon
    except Exception:
        return
    model_folds = fold_metrics[fold_metrics["model"].eq(best_model)]
    reference = model_folds[model_folds["feature_set"].eq(full_feature_set)][
        ["repeat", "fold", "AUPRC"]
    ].rename(columns={"AUPRC": "reference_AUPRC"})
    tests = []
    for feature_set in sorted(model_folds["feature_set"].unique()):
        comparison = model_folds[model_folds["feature_set"].eq(feature_set)][
            ["repeat", "fold", "AUPRC"]
        ]
        paired = reference.merge(comparison, on=["repeat", "fold"], how="inner").dropna()
        if feature_set == full_feature_set or paired.empty:
            statistic, p_value = 0.0, 1.0
        elif np.allclose(paired["reference_AUPRC"], paired["AUPRC"]):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = wilcoxon(paired["reference_AUPRC"], paired["AUPRC"])
        tests.append(
            {
                "comparison_type": "feature_ablation",
                "model": best_model,
                "reference_model": best_model,
                "comparison_model": best_model,
                "reference_feature_set": full_feature_set,
                "comparison_feature_set": feature_set,
                "metric": "AUPRC",
                "paired_folds": len(paired),
                "wilcoxon_statistic": float(statistic),
                "p_value_two_sided": float(p_value),
            }
        )
    full_feature_folds = fold_metrics[fold_metrics["feature_set"].eq(full_feature_set)]
    model_reference = full_feature_folds[full_feature_folds["model"].eq(best_model)][
        ["repeat", "fold", "AUPRC"]
    ].rename(columns={"AUPRC": "reference_AUPRC"})
    for comparison_model in sorted(full_feature_folds["model"].unique()):
        comparison = full_feature_folds[full_feature_folds["model"].eq(comparison_model)][
            ["repeat", "fold", "AUPRC"]
        ]
        paired = model_reference.merge(comparison, on=["repeat", "fold"], how="inner").dropna()
        if comparison_model == best_model or paired.empty:
            statistic, p_value = 0.0, 1.0
        elif np.allclose(paired["reference_AUPRC"], paired["AUPRC"]):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = wilcoxon(paired["reference_AUPRC"], paired["AUPRC"])
        tests.append(
            {
                "comparison_type": "model_comparison",
                "model": comparison_model,
                "reference_model": best_model,
                "comparison_model": comparison_model,
                "reference_feature_set": full_feature_set,
                "comparison_feature_set": full_feature_set,
                "metric": "AUPRC",
                "paired_folds": len(paired),
                "wilcoxon_statistic": float(statistic),
                "p_value_two_sided": float(p_value),
            }
        )
    tests_df = pd.DataFrame(tests)
    if not tests_df.empty:
        order = np.argsort(tests_df["p_value_two_sided"].to_numpy())
        adjusted = np.empty(len(tests_df), dtype=float)
        running_max = 0.0
        total = len(tests_df)
        for rank, index in enumerate(order):
            running_max = max(running_max, (total - rank) * tests_df.iloc[index]["p_value_two_sided"])
            adjusted[index] = min(running_max, 1.0)
        tests_df["p_value_holm"] = adjusted
    tests_df.to_csv(reports_dir / "statistical_tests.csv", index=False)


def _save_fold_predictions(df, y_true, y_pred, y_prob, path, target, repeat, fold) -> None:
    probabilities = np.asarray(y_prob)
    probability = probabilities if probabilities.ndim == 1 else probabilities.max(axis=1)
    data = {
        "repeat": repeat,
        "fold": fold,
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
