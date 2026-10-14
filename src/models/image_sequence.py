"""Raw-image RNN, LSTM, Transformer, and ViT classifiers."""

from __future__ import annotations

import torch
from torch import nn


class ImageRNNClassifier(nn.Module):
    """Read each grayscale image row as one sequence step."""

    def __init__(self, image_size: int = 224, n_classes: int = 2, hidden_size: int = 96, dropout: float = 0.1) -> None:
        super().__init__()
        self.rnn = nn.RNN(input_size=image_size, hidden_size=hidden_size, batch_first=True)
        self.head = _Head(hidden_size, n_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1)
        _, hidden = self.rnn(gray)
        return self.head(hidden[-1])


class ImageLSTMClassifier(nn.Module):
    """Read each grayscale image row with an LSTM."""

    def __init__(self, image_size: int = 224, n_classes: int = 2, hidden_size: int = 96, dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=image_size, hidden_size=hidden_size, batch_first=True)
        self.head = _Head(hidden_size, n_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1)
        _, (hidden, _) = self.lstm(gray)
        return self.head(hidden[-1])


class PatchTransformerClassifier(nn.Module):
    """Small transformer encoder over non-overlapping image patches."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        n_classes: int = 2,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        n_patches = (image_size // patch_size) ** 2
        self.position = nn.Parameter(torch.zeros(1, n_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = _Head(d_model, n_classes, dropout)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2) + self.position
        encoded = self.encoder(tokens)
        return self.head(encoded.mean(dim=1))


class TinyViTClassifier(nn.Module):
    """Compact ViT-style classifier with a class token."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        n_classes: int = 2,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        n_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = _Head(d_model, n_classes, dropout)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, patches], dim=1) + self.position
        encoded = self.encoder(tokens)
        return self.head(encoded[:, 0])


class _Head(nn.Module):
    def __init__(self, in_features: int, n_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, max(32, in_features // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, in_features // 2), n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
