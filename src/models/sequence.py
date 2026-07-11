"""Sequence-style neural classifiers for prepared feature vectors."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class FeatureRNNClassifier(nn.Module):
    """Treat a normalized feature vector as a sequence of scalar tokens."""

    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        d_model: int = 64,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tokenizer = _FeatureTokenizer(n_features, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.tokenizer.n_tokens, d_model))
        self.rnn = nn.RNN(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = _Head(hidden_size, n_classes, dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x) + self.position
        _, hidden = self.rnn(tokens)
        return self.head(hidden[-1])


class FeatureLSTMClassifier(nn.Module):
    """LSTM classifier over scalar feature tokens."""

    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        d_model: int = 64,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tokenizer = _FeatureTokenizer(n_features, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.tokenizer.n_tokens, d_model))
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = _Head(hidden_size, n_classes, dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x) + self.position
        _, (hidden, _) = self.lstm(tokens)
        return self.head(hidden[-1])


class FeatureTransformerClassifier(nn.Module):
    """Transformer encoder classifier over scalar feature tokens."""

    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        self.tokenizer = _FeatureTokenizer(n_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position = nn.Parameter(torch.zeros(1, self.tokenizer.n_tokens + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = _Head(d_model, n_classes, dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.position
        encoded = self.encoder(tokens)
        return self.head(encoded[:, 0])


class _FeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_model: int, max_tokens: int = 64) -> None:
        super().__init__()
        self.n_features = n_features
        self.token_size = max(1, math.ceil(n_features / max_tokens))
        self.n_tokens = math.ceil(n_features / self.token_size)
        self.padded_features = self.n_tokens * self.token_size
        self.projection = nn.Linear(self.token_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < self.padded_features:
            x = F.pad(x, (0, self.padded_features - x.shape[1]))
        tokens = x[:, : self.padded_features].reshape(x.shape[0], self.n_tokens, self.token_size)
        return self.projection(tokens)


class _Head(nn.Module):
    def __init__(self, in_features: int, n_classes: int, dropout: float) -> None:
        super().__init__()
        hidden = max(16, int(math.sqrt(in_features * max(n_classes, 2))) * 4)
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
