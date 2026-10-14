"""Training loop for raw-image neural baselines."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.data.dataset import ImageClassificationDataset
from src.data.targets import get_target_spec
from src.models.image_sequence import (
    ImageLSTMClassifier,
    ImageRNNClassifier,
    PatchTransformerClassifier,
    TinyViTClassifier,
)
from src.utils.io import ensure_dir
from src.utils.metrics import binary_metrics, multiclass_metrics
from src.utils.plotting import save_confusion_matrix, save_roc_curve


def train_raw_image_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: str | Path,
    task: str = "cancer",
    seed: int = 42,
    models: list[str] | None = None,
    image_size: int = 224,
    epochs: int = 8,
    batch_size: int = 16,
    lr: float = 1e-3,
    patience: int = 3,
    device: str | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Train raw-image RNN/LSTM/Transformer/ViT baselines."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    out_dir = Path(out_dir)
    ensure_dir(out_dir / "models")
    ensure_dir(out_dir / "predictions")
    ensure_dir(out_dir / "figures")
    torch.manual_seed(seed)
    np.random.seed(seed)
    resolved_device = _resolve_device(device)

    target = get_target_spec(task)
    label_col = target.label_column
    encoder = LabelEncoder()
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["_encoded_label"] = encoder.fit_transform(train_df[label_col].to_numpy())
    val_df["_encoded_label"] = encoder.transform(val_df[label_col].to_numpy())
    test_df["_encoded_label"] = encoder.transform(test_df[label_col].to_numpy())
    n_classes = len(encoder.classes_)

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_loader = DataLoader(
        ImageClassificationDataset(train_df, label_col="_encoded_label", transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=resolved_device == "cuda",
    )
    val_loader = DataLoader(
        ImageClassificationDataset(val_df, label_col="_encoded_label", transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=resolved_device == "cuda",
    )
    test_loader = DataLoader(
        ImageClassificationDataset(test_df, label_col="_encoded_label", transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=resolved_device == "cuda",
    )

    specs = {
        "raw_rnn": ImageRNNClassifier,
        "raw_lstm": ImageLSTMClassifier,
        "raw_transformer": PatchTransformerClassifier,
        "vit": TinyViTClassifier,
    }
    selected = models or list(specs)
    metrics: dict[str, dict] = {}
    rows = []
    y_train = train_df["_encoded_label"].to_numpy()

    for model_name in selected:
        if model_name not in specs:
            raise KeyError(f"Unknown raw image model: {model_name}")
        feature_name = f"pixels_{image_size}"
        started = time.perf_counter()
        print(
            f"[train_raw_image] Fitting {feature_name}/{model_name} on {resolved_device}: "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}",
            flush=True,
        )
        model = specs[model_name](image_size=image_size, n_classes=n_classes).to(resolved_device)
        model, best_val_score = _fit_one_model(
            model,
            train_loader,
            val_loader,
            y_train=y_train,
            n_classes=n_classes,
            task=task,
            epochs=epochs,
            lr=lr,
            patience=patience,
            device=resolved_device,
        )
        val_encoded, val_prob, _ = _predict(model, val_loader, resolved_device)
        test_encoded, test_prob, test_ids = _predict(model, test_loader, resolved_device)
        val_pred = encoder.inverse_transform(np.argmax(val_prob, axis=1))
        test_pred = encoder.inverse_transform(np.argmax(test_prob, axis=1))
        val_true = encoder.inverse_transform(val_encoded)
        test_true = encoder.inverse_transform(test_encoded)
        val_y_prob = _positive_or_matrix_probability(val_prob, task)
        test_y_prob = _positive_or_matrix_probability(test_prob, task)
        val_metrics = _score(val_true, val_pred, val_y_prob, task, class_labels=encoder.classes_)
        test_metrics = _score(test_true, test_pred, test_y_prob, task, class_labels=encoder.classes_)
        training_seconds = time.perf_counter() - started

        torch.save(
            {
                "model_name": model_name,
                "image_size": image_size,
                "n_classes": n_classes,
                "class_names": encoder.classes_.tolist(),
                "state_dict": model.cpu().state_dict(),
            },
            out_dir / "models" / f"{feature_name}_{model_name}.pt",
        )
        model.to(resolved_device)
        metrics[model_name] = {
            "feature_set": feature_name,
            "best_params": {
                "image_size": image_size,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "patience": patience,
            },
            "target": target.to_dict(),
            "imbalance_strategy": "class_weight",
            "training_seconds": training_seconds,
            "validation_selection_score": best_val_score,
            "validation": val_metrics,
            "test": test_metrics,
        }
        _save_predictions(
            test_ids,
            test_true,
            test_pred,
            test_y_prob,
            out_dir / "predictions" / f"{feature_name}_{model_name}_predictions.csv",
            test_df,
            target,
        )
        _save_eval_figures(feature_name, model_name, test_true, test_pred, test_y_prob, out_dir)
        rows.append(_leaderboard_row(feature_name, model_name, test_metrics, training_seconds))
        print(
            f"[train_raw_image] Finished {feature_name}/{model_name}: "
            f"best_val_score={best_val_score:.4f}, test_auc={test_metrics.get('ROC-AUC')}",
            flush=True,
        )

    leaderboard = pd.DataFrame(rows)
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values(
            by=["ROC-AUC", "balanced_accuracy"], ascending=False, na_position="last"
        )
    return metrics, leaderboard


def _fit_one_model(model, train_loader, val_loader, y_train, n_classes, task, epochs, lr, patience, device):
    import torch

    criterion = torch.nn.CrossEntropyLoss(weight=_class_weights(y_train, n_classes).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_score = -np.inf
    best_state = None
    misses = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += float(loss.item()) * int(labels.numel())
            seen += int(labels.numel())
        val_labels, val_prob, _ = _predict(model, val_loader, device)
        val_pred = np.argmax(val_prob, axis=1)
        score = _selection_score(
            val_labels,
            val_pred,
            _positive_or_matrix_probability(val_prob, task),
            task,
            class_labels=np.arange(n_classes),
        )
        if score > best_score:
            best_score = float(score)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            misses = 0
        else:
            misses += 1
        print(
            f"[train_raw_image]   epoch {epoch}/{epochs}: "
            f"loss={running_loss / max(seen, 1):.4f}, val_score={score:.4f}, best={best_score:.4f}",
            flush=True,
        )
        if misses >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_score


def _resolve_device(device: str | None) -> str:
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        print("[train_raw_image] CUDA was requested but is unavailable; falling back to CPU.", flush=True)
        return "cpu"
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def _predict(model, loader, device: str):
    import torch

    model.eval()
    labels = []
    probabilities = []
    image_ids = []
    with torch.no_grad():
        for images, batch_labels, batch_ids in loader:
            logits = model(images.to(device, non_blocking=True))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(batch_labels.numpy())
            image_ids.extend(batch_ids)
    return np.concatenate(labels), np.vstack(probabilities), image_ids


def _class_weights(y: np.ndarray, n_classes: int):
    import torch

    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (n_classes * counts)
    return torch.as_tensor(weights, dtype=torch.float32)


def _positive_or_matrix_probability(probabilities: np.ndarray, task: str) -> np.ndarray:
    if get_target_spec(task).is_binary and probabilities.shape[1] == 2:
        return probabilities[:, 1]
    return probabilities


def _selection_score(y_true, y_pred, y_prob, task: str, class_labels=None) -> float:
    metrics = _score(y_true, y_pred, y_prob, task, class_labels=class_labels)
    score = metrics.get("ROC-AUC" if get_target_spec(task).is_binary else "macro ROC-AUC", np.nan)
    if not np.isfinite(score):
        score = metrics.get("balanced_accuracy", -np.inf)
    return float(score)


def _score(y_true, y_pred, y_prob, task: str, class_labels=None) -> dict:
    target = get_target_spec(task)
    if target.is_binary:
        return binary_metrics(
            np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob), class_names=target.class_names
        )
    return multiclass_metrics(
        np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob), class_labels=class_labels
    )


def _save_predictions(image_ids, y_true, y_pred, y_prob, path: Path, df, target) -> None:
    probability = y_prob if np.asarray(y_prob).ndim == 1 else np.max(y_prob, axis=1)
    data = {
        "image_id": image_ids,
        "lesion_id": df["lesion_id"].to_numpy(),
        "dx": df["dx"].to_numpy(),
        "target": target.name,
        "true_label": y_true,
        "predicted_label": y_pred,
        "probability": probability,
    }
    if target.is_binary:
        data[f"probability_{target.class_names[1]}"] = probability
    pd.DataFrame(data).to_csv(path, index=False)


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


def _leaderboard_row(feature_set: str, model_name: str, metrics: dict, training_seconds: float) -> dict:
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
        "AUPRC": metrics.get("Average Precision"),
        "training_seconds": training_seconds,
        "macro F1": metrics.get("macro F1"),
        "weighted F1": metrics.get("weighted F1"),
        "macro ROC-AUC": metrics.get("macro ROC-AUC"),
    }
