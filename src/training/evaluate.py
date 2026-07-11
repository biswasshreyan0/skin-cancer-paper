"""Model prediction and metric evaluation helpers."""

from __future__ import annotations

import numpy as np

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


def evaluate_model(model, X, y, task: str = "binary") -> tuple[dict, np.ndarray, np.ndarray | None]:
    y_pred, y_prob = predict_labels_and_probabilities(model, X)
    if task == "binary":
        return binary_metrics(y, y_pred, y_prob), y_pred, y_prob
    return multiclass_metrics(y, y_pred, y_prob), y_pred, y_prob
