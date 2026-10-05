"""Model prediction and metric evaluation helpers."""

from __future__ import annotations

import numpy as np

from src.data.targets import get_target_spec
from src.utils.metrics import binary_metrics, multiclass_metrics


def predict_labels_and_probabilities(model, X):
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            y_prob = proba[:, 1]
        else:
            y_prob = proba
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        y_prob = 1.0 / (1.0 + np.exp(-scores))
    else:
        y_prob = None
    return y_pred, y_prob


def evaluate_model(model, X, y, task: str = "cancer") -> tuple[dict, np.ndarray, np.ndarray | None]:
    y_pred, y_prob = predict_labels_and_probabilities(model, X)
    target = get_target_spec(task)
    if target.is_binary:
        return binary_metrics(y, y_pred, y_prob, class_names=target.class_names), y_pred, y_prob
    # sklearn's probability columns follow ``classes_``.  Preserve that order
    # so each one-vs-rest AUPRC is assigned to the correct diagnosis.
    class_labels = getattr(model, "classes_", None)
    return multiclass_metrics(y, y_pred, y_prob, class_labels=class_labels), y_pred, y_prob
