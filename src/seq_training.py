"""Sequence dataset utilities and training helpers."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class TrajDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: Tuple[str, str] = ("label_X", "label_Y"),
    window_size: int = 20,
    session_col: str | None = None,
    target_mode: str = "abs",
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Create (N, T, D) sequences and (N, 2) targets from dataframe."""

    X_seqs: List[np.ndarray] = []
    y_seqs: List[np.ndarray] = []
    idx_seq: List[int] = []

    if session_col is not None:
        groups = df.groupby(session_col)
    else:
        groups = [(None, df)]

    for _, g in groups:
        if "t_ms" in g.columns:
            g = g.sort_values("t_ms", kind="mergesort")
        values = g[feature_cols].values
        y_values = g[list(target_cols)].values if set(target_cols).issubset(g.columns) else None

        for i in range(window_size - 1, len(g)):
            start = i - window_size + 1
            end = i + 1
            X_seqs.append(values[start:end])
            if y_values is not None:
                if target_mode == "delta":
                    if i == 0:
                        y_seqs.append(np.zeros((2,), dtype=float))
                    else:
                        y_seqs.append(y_values[i] - y_values[i - 1])
                else:
                    y_seqs.append(y_values[i])
            idx_seq.append(int(g.index[i]))

    X_arr = np.asarray(X_seqs)
    y_arr = np.asarray(y_seqs) if y_seqs else np.empty((0, 2))
    return X_arr, y_arr, idx_seq


def fit_seq_scaler(X_train_seqs: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler on flattened sequences."""

    n, t, d = X_train_seqs.shape
    X_flat = X_train_seqs.reshape(n * t, d)
    scaler = StandardScaler()
    scaler.fit(X_flat)
    return scaler


def transform_seq_scaler(X_seqs: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Scale sequences with a pre-fit scaler."""

    n, t, d = X_seqs.shape
    X_flat = X_seqs.reshape(n * t, d)
    X_scaled = scaler.transform(X_flat)
    return X_scaled.reshape(n, t, d)


def train_torch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    patience: int,
    lr: float,
    device: torch.device,
    ckpt_path,
) -> dict:
    """Train with early stopping and optional checkpointing."""

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "best_val": float("inf"), "best_epoch": -1}
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        if val_loss < history["best_val"]:
            history["best_val"] = float(val_loss)
            history["best_epoch"] = epoch
            epochs_no_improve = 0
            if ckpt_path is not None:
                torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    return history


def predict_torch_model(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict on a loader and return (y_true, y_pred)."""

    model.eval()
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch).cpu().numpy()
            all_pred.append(y_pred)
            all_true.append(y_batch.numpy())

    y_true = np.vstack(all_true) if all_true else np.empty((0, 2))
    y_pred = np.vstack(all_pred) if all_pred else np.empty((0, 2))
    return y_true, y_pred


def seq_preds_to_pointwise(
    y_pred_seq: np.ndarray,
    idx_seq: List[int],
    n_total: int,
    *,
    agg: str = "mean",
    fill_value: float = np.nan,
) -> np.ndarray:
    """Map sequence predictions back to pointwise indices."""

    out = np.full((n_total, 2), fill_value, dtype=float)
    counts = np.zeros(n_total, dtype=float)

    for pred, idx in zip(y_pred_seq, idx_seq):
        if idx >= n_total:
            continue
        if np.isnan(out[idx]).any():
            out[idx] = pred
            counts[idx] = 1.0
        else:
            out[idx] += pred
            counts[idx] += 1.0

    if agg == "mean":
        mask = counts > 0
        out[mask] = out[mask] / counts[mask][:, None]

    return out
