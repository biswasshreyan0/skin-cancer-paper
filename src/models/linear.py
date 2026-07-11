"""Linear model helpers."""

from src.models.registry import get_model_registry


def make_logistic_regression(seed: int = 42):
    return get_model_registry(seed)["logistic_regression"].estimator
