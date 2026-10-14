"""Model registry for tabular ABCD/embedding experiments."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


@dataclass
class ModelSpec:
    name: str
    estimator: object
    param_grid: dict[str, list]
    optional: bool = False


def get_model_registry(seed: int = 42, use_class_weights: bool = True) -> dict[str, ModelSpec]:
    class_weight = "balanced" if use_class_weights else None
    registry: dict[str, ModelSpec] = {
        "decision_tree": ModelSpec(
            name="decision_tree",
            estimator=DecisionTreeClassifier(class_weight=class_weight, random_state=seed),
            param_grid={"max_depth": [2, 3, 4, 5, 10, 15], "min_samples_leaf": [5, 10, 20]},
        ),
        "logistic_regression": ModelSpec(
            name="logistic_regression",
            estimator=LogisticRegression(class_weight=class_weight, max_iter=5000, random_state=seed),
            param_grid={"C": [0.1, 1.0, 10.0]},
        ),
        "knn": ModelSpec(
            name="knn",
            estimator=KNeighborsClassifier(),
            param_grid={"n_neighbors": [3, 5, 7, 11]},
        ),
        "random_forest": ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(
                n_estimators=300, class_weight=class_weight, random_state=seed, n_jobs=-1
            ),
            param_grid={"max_depth": [None, 5, 10]},
        ),
    }

    try:
        from xgboost import XGBClassifier

        registry["xgboost"] = ModelSpec(
            name="xgboost",
            estimator=XGBClassifier(
                n_estimators=300,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=seed,
                n_jobs=-1,
            ),
            param_grid={"max_depth": [3, 4], "learning_rate": [0.03, 0.05]},
            optional=True,
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier

        registry["lightgbm"] = ModelSpec(
            name="lightgbm",
            estimator=LGBMClassifier(
                n_estimators=300,
                class_weight=class_weight,
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            ),
            param_grid={"max_depth": [-1, 5], "learning_rate": [0.03, 0.05]},
            optional=True,
        )
    except Exception:
        pass

    try:
        from catboost import CatBoostClassifier

        catboost_weight_args = {"auto_class_weights": "Balanced"} if use_class_weights else {}
        registry["catboost"] = ModelSpec(
            name="catboost",
            estimator=CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
                **catboost_weight_args,
            ),
            param_grid={"depth": [4, 6], "learning_rate": [0.03, 0.05], "iterations": [300]},
            optional=True,
        )
    except Exception:
        pass

    return registry
