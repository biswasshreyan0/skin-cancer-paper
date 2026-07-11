"""Boosting model helpers."""

from src.models.registry import get_model_registry


def make_boosting_models(seed: int = 42):
    registry = get_model_registry(seed)
    return {name: spec.estimator for name, spec in registry.items() if name in {"xgboost", "lightgbm"}}
