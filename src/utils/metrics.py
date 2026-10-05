"""Classification metrics for binary and multiclass experiments."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    class_names: tuple[str, str] | list[str] = ("negative", "positive"),
) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": cm.tolist(),
        "per_class": _per_class_metrics(
            y_true,
            y_pred,
            y_prob,
            class_labels=[0, 1],
            class_names=class_names,
        ),
    }
    if y_prob is not None:
        try:
            metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["ROC-AUC"] = float("nan")
        try:
            metrics["Average Precision"] = average_precision_score(y_true, y_prob)
            metrics["AUPRC"] = metrics["Average Precision"]
        except ValueError:
            metrics["Average Precision"] = float("nan")
            metrics["AUPRC"] = float("nan")
    else:
        metrics["ROC-AUC"] = float("nan")
        metrics["Average Precision"] = float("nan")
        metrics["AUPRC"] = float("nan")
    return metrics


def multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    class_labels: np.ndarray | list | None = None,
) -> dict:
    """Return aggregate and one-vs-rest metrics for every diagnosis class.

    ``class_labels`` must use the same order as the columns of a multiclass
    probability matrix.  Supplying it preserves classes that are absent from a
    particular evaluation split while still reporting them with zero-valued
    classification metrics and an undefined AUPRC.
    """
    if class_labels is None:
        class_labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)])).tolist()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "per_class": _per_class_metrics(y_true, y_pred, y_prob, class_labels=class_labels),
    }
    if y_prob is not None:
        try:
            metrics["macro ROC-AUC"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except ValueError:
            metrics["macro ROC-AUC"] = float("nan")
    else:
        metrics["macro ROC-AUC"] = float("nan")
    per_class_auprc = [
        values["AUPRC"] for values in metrics["per_class"].values() if np.isfinite(values["AUPRC"])
    ]
    metrics["macro AUPRC"] = float(np.mean(per_class_auprc)) if per_class_auprc else float("nan")
    return metrics


def _per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    class_labels: np.ndarray | list,
    class_names: np.ndarray | list | None = None,
) -> dict[str, dict[str, float]]:
    """Calculate precision, recall, F1, and one-vs-rest AUPRC by class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(class_labels)
    names = list(class_names) if class_names is not None else labels
    if len(labels) != len(names):
        raise ValueError("class_names and class_labels must have the same length.")

    probabilities = None if y_prob is None else np.asarray(y_prob)
    if probabilities is not None and probabilities.ndim == 1:
        if len(labels) != 2:
            raise ValueError("A one-dimensional probability array requires exactly two classes.")
        # Binary classifiers conventionally return P(class_labels[1]).
        probabilities = np.column_stack([1.0 - probabilities, probabilities])
    if probabilities is not None and (
        probabilities.ndim != 2 or probabilities.shape[1] != len(labels)
    ):
        raise ValueError("Probability columns must match class_labels.")

    result: dict[str, dict[str, float]] = {}
    for index, (label, name) in enumerate(zip(labels, names, strict=True)):
        is_class = y_true == label
        predicted_class = y_pred == label
        precision = precision_score(is_class, predicted_class, zero_division=0)
        recall = recall_score(is_class, predicted_class, zero_division=0)
        f1 = f1_score(is_class, predicted_class, zero_division=0)
        if probabilities is None or not np.any(is_class):
            auprc = float("nan")
        else:
            auprc = average_precision_score(is_class, probabilities[:, index])
        result[str(name)] = {
            "precision": float(precision),
            "recall": float(recall),
            "F1": float(f1),
            "AUPRC": float(auprc),
        }
    return result
