"""Decision tree model helpers."""

from src.models.registry import get_model_registry


def make_decision_tree(seed: int = 42):
    return get_model_registry(seed)["decision_tree"].estimator
