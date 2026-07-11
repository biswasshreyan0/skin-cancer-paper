"""Forest model helpers."""

from src.models.registry import get_model_registry


def make_random_forest(seed: int = 42):
    return get_model_registry(seed)["random_forest"].estimator
