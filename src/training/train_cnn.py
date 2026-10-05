"""Optional ResNet18 CNN baseline."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import ImageClassificationDataset
from src.data.targets import get_target_spec
from src.models.cnn import make_resnet18_binary
from src.utils.io import ensure_dir
from src.utils.metrics import binary_metrics


def train_cnn_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: str | Path,
    batch_size: int = 8,
    epochs: int = 5,
    lr: float = 1e-3,
    seed: int = 42,
    pretrained: bool = True,
    task: str = "cancer",
    device: str | None = None,
) -> dict:
    """Train a lightweight frozen-backbone ResNet18 baseline and evaluate on test."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    target = get_target_spec(task)
    if not target.is_binary:
        raise ValueError("The ResNet18 baseline currently supports binary targets only.")
    if device == "cuda" and not torch.cuda.is_available():
        print("[train_cnn] CUDA requested but unavailable; falling back to CPU.", flush=True)
        device = "cpu"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    out_dir = Path(out_dir)
    ensure_dir(out_dir / "models")
    ensure_dir(out_dir / "predictions")

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    label_col = target.label_column
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_loader = DataLoader(
        ImageClassificationDataset(train_df, label_col=label_col, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        ImageClassificationDataset(val_df, label_col=label_col, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        ImageClassificationDataset(test_df, label_col=label_col, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"[train_cnn] device={device}, train={len(train_df)}, val={len(val_df)}, test={len(test_df)}", flush=True)
    model = make_resnet18_binary(pretrained=pretrained).to(device)
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=lr)
    counts = np.bincount(train_df[label_col].to_numpy(dtype=int), minlength=2).astype(float)
    weights = counts.sum() / (2.0 * np.maximum(counts, 1.0))
    criterion = torch.nn.CrossEntropyLoss(weight=torch.as_tensor(weights, dtype=torch.float32, device=device))
    best_val_auc = -np.inf
    best_path = out_dir / "models" / "cnn_resnet18_best.pt"

    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * int(labels.numel())
            seen += int(labels.numel())

        y_val, p_val = _predict_cnn(model, val_loader, device)
        try:
            from sklearn.metrics import roc_auc_score

            val_auc = roc_auc_score(y_val, p_val)
        except Exception:
            val_auc = 0.0
        if val_auc > best_val_auc:
            best_val_auc = float(val_auc)
            torch.save(model.state_dict(), best_path)
        mean_loss = running_loss / max(seen, 1)
        print(
            f"[train_cnn] epoch {epoch}/{epochs}: train_loss={mean_loss:.4f}, "
            f"val_auc={val_auc:.4f}, best_val_auc={best_val_auc:.4f}",
            flush=True,
        )

    model.load_state_dict(torch.load(best_path, map_location=device))
    y_true, y_prob, image_ids = _predict_cnn(model, test_loader, device, return_ids=True)
    y_pred = (y_prob >= 0.5).astype(int)
    training_seconds = time.perf_counter() - started
    metrics = binary_metrics(y_true, y_pred, y_prob, class_names=target.class_names)
    pd.DataFrame(
        {
            "image_id": image_ids,
            "lesion_id": test_df["lesion_id"].to_numpy(),
            "dx": test_df["dx"].to_numpy(),
            "target": target.name,
            "true_label": y_true,
            "predicted_label": y_pred,
            "probability": y_prob,
            f"probability_{target.class_names[1]}": y_prob,
        }
    ).to_csv(out_dir / "predictions" / "pixels_224_cnn_resnet18_predictions.csv", index=False)
    return {
        "target": target.to_dict(),
        "imbalance_strategy": "class_weight",
        "image_size": 224,
        "training_seconds": training_seconds,
        "validation_best_ROC-AUC": best_val_auc,
        "test": metrics,
    }


def _predict_cnn(model, loader, device: str, return_ids: bool = False):
    import torch

    model.eval()
    y_true = []
    y_prob = []
    image_ids = []
    with torch.no_grad():
        for images, labels, ids in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            y_prob.extend(probs)
            y_true.extend(labels.numpy())
            image_ids.extend(ids)
    if return_ids:
        return np.asarray(y_true), np.asarray(y_prob), image_ids
    return np.asarray(y_true), np.asarray(y_prob)
