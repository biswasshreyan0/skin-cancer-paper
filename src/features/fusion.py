"""Build leakage-safe feature sets for tabular experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.abcde import ABCDE_FEATURE_COLUMNS


@dataclass
class FeatureSet:
    name: str
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    feature_names: list[str]
    transformer: object | None = None


def build_feature_sets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    abcde_df: pd.DataFrame,
    embedding_cache: dict[str, np.ndarray],
    out_dir: str | Path,
    pca_dim: int = 64,
) -> dict[str, FeatureSet]:
    """Create all requested feature sets, fitting transforms on train only."""
    abcde_by_id = abcde_df.copy()
    abcde_by_id["image_id"] = abcde_by_id["image_id"].astype(str)
    abcde_by_id = abcde_by_id.set_index("image_id")
    emb_by_id = _embedding_lookup(embedding_cache)

    train_abcde = _abcde_matrix(train_df, abcde_by_id)
    val_abcde = _abcde_matrix(val_df, abcde_by_id)
    test_abcde = _abcde_matrix(test_df, abcde_by_id)
    train_emb = _embedding_matrix(train_df, emb_by_id)
    val_emb = _embedding_matrix(val_df, emb_by_id)
    test_emb = _embedding_matrix(test_df, emb_by_id)
    train_raw25 = _raw_pixel_matrix(train_df)
    val_raw25 = _raw_pixel_matrix(val_df)
    test_raw25 = _raw_pixel_matrix(test_df)

    feature_sets: dict[str, FeatureSet] = {}

    raw25_pipeline = Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())])
    raw25_names = [f"raw25_px_{i:03d}" for i in range(train_raw25.shape[1])]
    feature_sets["raw25_pixels"] = _fit_tabular_set(
        "raw25_pixels", raw25_pipeline, train_raw25, val_raw25, test_raw25, raw25_names
    )

    abcde_pipeline = Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())])
    feature_sets["abcde"] = _fit_tabular_set(
        "abcde", abcde_pipeline, train_abcde, val_abcde, test_abcde, ABCDE_FEATURE_COLUMNS
    )

    emb_pipeline = Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())])
    embedding_names = [f"emb_{i:04d}" for i in range(train_emb.shape[1])]
    feature_sets["embeddings"] = _fit_tabular_set(
        "embeddings", emb_pipeline, train_emb, val_emb, test_emb, embedding_names
    )

    combined_train = np.hstack([train_abcde, train_emb])
    combined_val = np.hstack([val_abcde, val_emb])
    combined_test = np.hstack([test_abcde, test_emb])
    combined_names = ABCDE_FEATURE_COLUMNS + embedding_names
    combined_pipeline = Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())])
    feature_sets["abcde_embeddings"] = _fit_tabular_set(
        "abcde_embeddings", combined_pipeline, combined_train, combined_val, combined_test, combined_names
    )

    pca_dim = max(1, min(pca_dim, train_emb.shape[0], train_emb.shape[1]))
    pca_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_dim, random_state=42)),
        ]
    )
    pca_pipeline.fit(train_emb)
    train_pca = pca_pipeline.transform(train_emb)
    val_pca = pca_pipeline.transform(val_emb)
    test_pca = pca_pipeline.transform(test_emb)
    pca_names = [f"emb_pca_{i:03d}" for i in range(train_pca.shape[1])]
    abcde_pca_pipeline = Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())])
    feature_sets["abcde_pca_embeddings"] = _fit_tabular_set(
        "abcde_pca_embeddings",
        abcde_pca_pipeline,
        np.hstack([train_abcde, train_pca]),
        np.hstack([val_abcde, val_pca]),
        np.hstack([test_abcde, test_pca]),
        ABCDE_FEATURE_COLUMNS + pca_names,
    )
    feature_sets["abcde_pca_embeddings"].transformer = {
        "embedding_pca": pca_pipeline,
        "final_preprocessor": feature_sets["abcde_pca_embeddings"].transformer,
    }

    meta_processor = ColumnTransformer(
        transformers=[
            ("age", Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())]), ["age"]),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _dense_one_hot_encoder()),
                    ]
                ),
                ["sex", "localization"],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    meta_processor.fit(train_df[["age", "sex", "localization"]])
    train_meta = meta_processor.transform(train_df[["age", "sex", "localization"]])
    val_meta = meta_processor.transform(val_df[["age", "sex", "localization"]])
    test_meta = meta_processor.transform(test_df[["age", "sex", "localization"]])
    meta_names = list(meta_processor.get_feature_names_out())
    metadata_only_pipeline = Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())])
    feature_sets["metadata"] = _fit_tabular_set(
        "metadata", metadata_only_pipeline, train_meta, val_meta, test_meta, meta_names
    )

    metadata_train = np.hstack([combined_train, train_meta])
    metadata_val = np.hstack([combined_val, val_meta])
    metadata_test = np.hstack([combined_test, test_meta])
    metadata_pipeline = Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())])
    feature_sets["abcde_embeddings_metadata"] = _fit_tabular_set(
        "abcde_embeddings_metadata",
        metadata_pipeline,
        metadata_train,
        metadata_val,
        metadata_test,
        combined_names + meta_names,
    )
    feature_sets["abcde_embeddings_metadata"].transformer = {
        "metadata_processor": meta_processor,
        "final_preprocessor": feature_sets["abcde_embeddings_metadata"].transformer,
    }

    artifacts_dir = Path(out_dir) / "models" / "feature_transforms"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for name, fs in feature_sets.items():
        if fs.transformer is not None:
            joblib.dump(fs.transformer, artifacts_dir / f"{name}_transformer.joblib")
    return feature_sets


def _fit_tabular_set(
    name: str,
    transformer: Pipeline,
    X_train_raw: np.ndarray,
    X_val_raw: np.ndarray,
    X_test_raw: np.ndarray,
    feature_names: list[str],
) -> FeatureSet:
    transformer.fit(X_train_raw)
    return FeatureSet(
        name=name,
        X_train=transformer.transform(X_train_raw),
        X_val=transformer.transform(X_val_raw),
        X_test=transformer.transform(X_test_raw),
        feature_names=feature_names,
        transformer=transformer,
    )


def _dense_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _embedding_lookup(cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    image_ids = [str(x) for x in cache["image_ids"].tolist()]
    embeddings = cache["embeddings"]
    return {image_id: embeddings[i] for i, image_id in enumerate(image_ids)}


def _embedding_matrix(df: pd.DataFrame, emb_by_id: dict[str, np.ndarray]) -> np.ndarray:
    return np.vstack([emb_by_id[str(image_id)] for image_id in df["image_id"]])


def _abcde_matrix(df: pd.DataFrame, abcde_by_id: pd.DataFrame) -> np.ndarray:
    return abcde_by_id.loc[df["image_id"].astype(str), ABCDE_FEATURE_COLUMNS].to_numpy(dtype=float)


def _raw_pixel_matrix(df: pd.DataFrame, size: int = 25) -> np.ndarray:
    rows = []
    for path in df["image_path"]:
        image = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
        rows.append(np.asarray(image, dtype=np.float32).reshape(-1) / 255.0)
    return np.vstack(rows)
