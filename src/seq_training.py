"""Sequence dataset utilities and training helpers."""

from __future__ import annotations

from typing import Dict, List, Tuple

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
    segment_col: str | None = None,
    target_mode: str = "abs",
    time_col: str = "t_ms",
    max_dt_ms: float | None = None,
    max_window_ms: float | None = None,
    enforce_time_checks: bool = True,
    return_stats: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[int]] | Tuple[np.ndarray, np.ndarray, List[int], Dict[str, float]]:
    """Create (N, T, D) sequences and (N, 2) targets from dataframe."""

    X_seqs: List[np.ndarray] = []
    y_seqs: List[np.ndarray] = []
    idx_seq: List[int] = []
    stats: Dict[str, float] = {
        "time_checks_enabled": float(bool(enforce_time_checks)),
        "n_groups": 0.0,
        "n_rows": float(len(df)),
        "n_candidate_windows": 0.0,
        "n_kept_windows": 0.0,
        "rejected_empty_dt": 0.0,
        "rejected_non_finite_dt": 0.0,
        "rejected_non_positive_dt": 0.0,
        "rejected_max_dt": 0.0,
        "rejected_max_window": 0.0,
    }

    if session_col is not None and segment_col is not None and segment_col in df.columns:
        groups = df.groupby([session_col, segment_col])
    elif session_col is not None:
        groups = df.groupby(session_col)
    else:
        groups = [(None, df)]

    for _, g in groups:
        stats["n_groups"] += 1.0
        if time_col in g.columns:
            g = g.sort_values(time_col, kind="mergesort")
        values = g[feature_cols].values
        y_values = g[list(target_cols)].values if set(target_cols).issubset(g.columns) else None
        t_values = g[time_col].to_numpy(dtype=float) if time_col in g.columns else None
        stats["n_candidate_windows"] += float(max(len(g) - window_size + 1, 0))

        for i in range(window_size - 1, len(g)):
            start = i - window_size + 1
            end = i + 1
            if enforce_time_checks and t_values is not None:
                t_win = t_values[start:end]
                dt = np.diff(t_win)
                if dt.size == 0:
                    stats["rejected_empty_dt"] += 1.0
                    continue
                if not np.all(np.isfinite(dt)):
                    stats["rejected_non_finite_dt"] += 1.0
                    continue
                if np.any(dt <= 0):
                    stats["rejected_non_positive_dt"] += 1.0
                    continue
                if max_dt_ms is not None and np.max(dt) >= max_dt_ms:
                    stats["rejected_max_dt"] += 1.0
                    continue
                if max_window_ms is not None and (t_win[-1] - t_win[0]) >= max_window_ms:
                    stats["rejected_max_window"] += 1.0
                    continue
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
    stats["n_kept_windows"] = float(len(idx_seq))
    stats["n_rejected_windows"] = float(stats["n_candidate_windows"] - stats["n_kept_windows"])
    stats["acceptance_ratio"] = (
        float(stats["n_kept_windows"] / max(1.0, stats["n_candidate_windows"]))
        if stats["n_candidate_windows"] > 0
        else 0.0
    )

    if return_stats:
        return X_arr, y_arr, idx_seq, stats
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
    target_weights: np.ndarray | None = None,
) -> dict:
    """Train with early stopping and optional checkpointing."""

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    weight_t = None
    if target_weights is not None:
        w = np.asarray(target_weights, dtype=np.float32).reshape(1, -1)
        if w.shape[1] != 2:
            raise ValueError("target_weights must have shape (2,)")
        weight_t = torch.tensor(w, dtype=torch.float32, device=device)

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
            if weight_t is None:
                loss = torch.mean((y_pred - y_batch) ** 2)
            else:
                loss = torch.mean(((y_pred - y_batch) ** 2) * weight_t)
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
                if weight_t is None:
                    loss = torch.mean((y_pred - y_batch) ** 2)
                else:
                    loss = torch.mean(((y_pred - y_batch) ** 2) * weight_t)
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
