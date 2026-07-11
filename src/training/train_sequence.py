"""Training loop for RNN, LSTM, and Transformer feature-sequence baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.features.fusion import FeatureSet
from src.models.sequence import FeatureLSTMClassifier, FeatureRNNClassifier, FeatureTransformerClassifier
from src.utils.io import ensure_dir
from src.utils.metrics import binary_metrics, multiclass_metrics
from src.utils.plotting import save_confusion_matrix, save_roc_curve


def train_sequence_models(
    feature_sets: dict[str, FeatureSet],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: str | Path,
    task: str = "binary",
    seed: int = 42,
    feature_set_names: list[str] | None = None,
    epochs: int = 25,
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 5,
    device: str | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Train neural sequence baselines on selected prepared feature sets."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    out_dir = Path(out_dir)
    ensure_dir(out_dir / "models")
    ensure_dir(out_dir / "predictions")
    ensure_dir(out_dir / "figures")
    torch.manual_seed(seed)
    np.random.seed(seed)
    resolved_device = _resolve_device(device)

    label_col = "binary_label" if task == "binary" else "multiclass_label"
    y_train_raw = train_df[label_col].to_numpy()
    y_val_raw = val_df[label_col].to_numpy()
    y_test_raw = test_df[label_col].to_numpy()
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)
    y_val = encoder.transform(y_val_raw)
    y_test = encoder.transform(y_test_raw)
    n_classes = len(encoder.classes_)
    selected_features = feature_set_names or ["abcde_embeddings"]

    specs = {
        "rnn": FeatureRNNClassifier,
        "lstm": FeatureLSTMClassifier,
        "transformer": FeatureTransformerClassifier,
    }
    metrics: dict[str, dict] = {}
    rows = []

    for feature_name in selected_features:
        if feature_name not in feature_sets:
            raise KeyError(f"Unknown feature set for sequence model: {feature_name}")
        feature_set = feature_sets[feature_name]
        metrics[feature_name] = {}
        loaders = _make_loaders(feature_set, y_train, y_val, y_test, batch_size)
        for model_name, model_cls in specs.items():
            print(
                f"[train_sequence] Fitting {feature_name}/{model_name} on {resolved_device}: "
                f"train={feature_set.X_train.shape}, val={feature_set.X_val.shape}, test={feature_set.X_test.shape}",
                flush=True,
            )
            model, best_val_score = _fit_one_model(
                model_cls=model_cls,
                n_features=feature_set.X_train.shape[1],
                n_classes=n_classes,
                train_loader=loaders["train"],
                val_loader=loaders["val"],
                y_train=y_train,
                task=task,
                epochs=epochs,
                lr=lr,
                patience=patience,
                device=resolved_device,
            )
            _, test_prob_encoded = _predict(model, loaders["test"], resolved_device)
            y_prob = _positive_or_matrix_probability(test_prob_encoded, task)
            y_pred_encoded = np.argmax(test_prob_encoded, axis=1)
            y_pred = encoder.inverse_transform(y_pred_encoded)
            y_true = y_test_raw
            test_metrics = _score(y_true, y_pred, y_prob, task)
            val_labels, val_prob_encoded = _predict(model, loaders["val"], resolved_device)
            val_pred = encoder.inverse_transform(np.argmax(val_prob_encoded, axis=1))
            val_true = encoder.inverse_transform(val_labels)
            val_metrics = _score(val_true, val_pred, _positive_or_matrix_probability(val_prob_encoded, task), task)

            model_path = out_dir / "models" / f"{feature_name}_{model_name}.pt"
            torch.save(
                {
                    "model_name": model_name,
                    "feature_set": feature_name,
                    "n_features": feature_set.X_train.shape[1],
                    "n_classes": n_classes,
                    "class_names": encoder.classes_.tolist(),
                    "state_dict": model.cpu().state_dict(),
                },
                model_path,
            )
            model.to(resolved_device)
            metrics[feature_name][model_name] = {
                "best_params": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "patience": patience,
                },
                "validation_selection_score": best_val_score,
                "validation": val_metrics,
                "test": test_metrics,
            }
            _save_predictions(
                test_df,
                y_true,
                y_pred,
                y_prob,
                out_dir / "predictions" / f"{feature_name}_{model_name}_predictions.csv",
            )
            _save_eval_figures(feature_name, model_name, y_true, y_pred, y_prob, out_dir)
            rows.append(_leaderboard_row(feature_name, model_name, test_metrics))
            print(
                f"[train_sequence] Finished {feature_name}/{model_name}: "
                f"best_val_score={best_val_score:.4f}, test_auc={test_metrics.get('ROC-AUC')}",
                flush=True,
            )

    leaderboard = pd.DataFrame(rows)
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values(
            by=["ROC-AUC", "balanced_accuracy"], ascending=False, na_position="last"
        )
    return metrics, leaderboard


def _fit_one_model(
    model_cls,
    n_features: int,
    n_classes: int,
    train_loader,
    val_loader,
    y_train: np.ndarray,
    task: str,
    epochs: int,
    lr: float,
    patience: int,
    device: str,
):
    import torch

    model = model_cls(n_features=n_features, n_classes=n_classes).to(device)
    class_weights = _class_weights(y_train, n_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_score = -np.inf
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        val_labels, val_prob = _predict(model, val_loader, device)
        val_pred = np.argmax(val_prob, axis=1)
        score = _selection_score(val_labels, val_pred, _positive_or_matrix_probability(val_prob, task), task)
        if score > best_score:
            best_score = float(score)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(
            f"[train_sequence]   epoch {epoch}/{epochs}: val_score={score:.4f}, best={best_score:.4f}",
            flush=True,
        )
        if epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_score


def _resolve_device(device: str | None) -> str:
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        print("[train_sequence] CUDA was requested but is unavailable; falling back to CPU.", flush=True)
        return "cpu"
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def _make_loaders(feature_set: FeatureSet, y_train, y_val, y_test, batch_size: int) -> dict:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    def dataset(X, y):
        return TensorDataset(
            torch.as_tensor(np.asarray(X), dtype=torch.float32),
            torch.as_tensor(np.asarray(y), dtype=torch.long),
        )

    return {
        "train": DataLoader(dataset(feature_set.X_train, y_train), batch_size=batch_size, shuffle=True),
        "val": DataLoader(dataset(feature_set.X_val, y_val), batch_size=batch_size, shuffle=False),
        "test": DataLoader(dataset(feature_set.X_test, y_test), batch_size=batch_size, shuffle=False),
    }


def _predict(model, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    labels = []
    probabilities = []
    with torch.no_grad():
        for features, batch_labels in loader:
            logits = model(features.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(batch_labels.numpy())
    return np.concatenate(labels), np.vstack(probabilities)


def _class_weights(y: np.ndarray, n_classes: int):
    import torch

    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (n_classes * counts)
    return torch.as_tensor(weights, dtype=torch.float32)


def _positive_or_matrix_probability(probabilities: np.ndarray, task: str) -> np.ndarray:
    if task == "binary" and probabilities.shape[1] == 2:
        return probabilities[:, 1]
    return probabilities


def _selection_score(y_true, y_pred, y_prob, task: str) -> float:
    metrics = _score(y_true, y_pred, y_prob, task)
    if task == "binary":
        score = metrics.get("ROC-AUC", np.nan)
    else:
        score = metrics.get("macro ROC-AUC", np.nan)
    if not np.isfinite(score):
        score = metrics.get("balanced_accuracy", -np.inf)
    return float(score)


def _score(y_true, y_pred, y_prob, task: str) -> dict:
    if task == "binary":
        return binary_metrics(np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob))
    return multiclass_metrics(np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob))


def _save_predictions(df, y_true, y_pred, y_prob, path: Path) -> None:
    if np.asarray(y_prob).ndim == 1:
        probability = y_prob
    else:
        probability = np.max(y_prob, axis=1)
    pd.DataFrame(
        {
            "image_id": df["image_id"].to_numpy(),
            "true_label": y_true,
            "predicted_label": y_pred,
            "probability": probability,
        }
    ).to_csv(path, index=False)


def _save_eval_figures(feature_set, model_name, y_true, y_pred, y_prob, out_dir: Path) -> None:
    try:
        save_confusion_matrix(y_true, y_pred, out_dir / "figures" / f"{feature_set}_{model_name}_confusion.png")
    except Exception:
        pass
    if np.asarray(y_prob).ndim == 1:
        try:
            save_roc_curve(y_true, y_prob, out_dir / "figures" / f"{feature_set}_{model_name}_roc.png")
        except Exception:
            pass


def _leaderboard_row(feature_set: str, model_name: str, metrics: dict) -> dict:
    return {
        "feature_set": feature_set,
        "model": model_name,
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "sensitivity": metrics.get("sensitivity"),
        "specificity": metrics.get("specificity"),
        "precision": metrics.get("precision"),
        "F1": metrics.get("F1"),
        "ROC-AUC": metrics.get("ROC-AUC"),
        "Average Precision": metrics.get("Average Precision"),
        "macro F1": metrics.get("macro F1"),
        "weighted F1": metrics.get("weighted F1"),
        "macro ROC-AUC": metrics.get("macro ROC-AUC"),
    }
