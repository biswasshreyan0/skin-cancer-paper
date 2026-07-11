"""KNN model helpers."""

from src.models.registry import get_model_registry


def make_knn(seed: int = 42):
    return get_model_registry(seed)["knn"].estimator
