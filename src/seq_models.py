"""Torch sequence models."""

from __future__ import annotations

import torch
import torch.nn as nn


def _diamond_head(
    *,
    in_dim: int,
    out_dim: int = 2,
    widen_factor: float = 2.0,
    min_small_dim: int = 8,
    dropout: float = 0.05,
) -> nn.Sequential:
    """Diamond MLP: n -> wider -> n -> smaller -> small -> out."""

    n = max(4, int(in_dim))
    wider = max(n + 4, int(round(n * widen_factor)))
    smaller = max(min_small_dim, n // 2)
    small = max(4, smaller // 2)

    return nn.Sequential(
        nn.Linear(n, wider),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(wider, n),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(n, smaller),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(smaller, small),
        nn.ReLU(),
        nn.Linear(small, out_dim),
    )


class LSTMRegressorDiamond(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout_lstm: float = 0.1,
        dropout_fc: float = 0.05,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_lstm if num_layers > 1 else 0.0,
        )

        # Bridge hidden state -> n_features, then strict diamond head.
        self.bridge = nn.Sequential(
            nn.Linear(hidden_dim, self.input_dim),
            nn.ReLU(),
            nn.Dropout(dropout_fc),
        )
        self.fc = _diamond_head(in_dim=self.input_dim, out_dim=2, dropout=dropout_fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        z = self.bridge(last_hidden)
        return self.fc(z)


class GRURegressorDiamond(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout_gru: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_gru if num_layers > 1 else 0.0,
        )

        self.bridge = nn.Sequential(
            nn.Linear(hidden_dim, self.input_dim),
            nn.ReLU(),
            nn.Dropout(dropout_gru),
        )
        self.fc = _diamond_head(in_dim=self.input_dim, out_dim=2, dropout=dropout_gru)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]
        z = self.bridge(last_hidden)
        return self.fc(z)
