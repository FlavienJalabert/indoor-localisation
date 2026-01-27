"""Metrics for 2D localization."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str,
    *,
    thresholds: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0),
) -> Dict:
    """Compute 2D localization metrics and coverage ratios."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape or y_true.ndim != 2 or y_true.shape[1] != 2:
        raise ValueError("y_true and y_pred must be (N,2)")

    ex = y_pred[:, 0] - y_true[:, 0]
    ey = y_pred[:, 1] - y_true[:, 1]
    err = np.sqrt(ex**2 + ey**2)

    rmse_x = np.sqrt(np.mean(ex**2))
    rmse_y = np.sqrt(np.mean(ey**2))
    rmse_2d = np.sqrt(np.mean(err**2))

    mae_x = np.mean(np.abs(ex))
    mae_y = np.mean(np.abs(ey))
    mae_2d = np.mean(err)

    bias_x = np.mean(ex)
    bias_y = np.mean(ey)

    r50 = np.percentile(err, 50)
    r68 = np.percentile(err, 68)
    r90 = np.percentile(err, 90)
    r95 = np.percentile(err, 95)
    r99 = np.percentile(err, 99)

    cov = {f"p_err_lt_{t}m": float(np.mean(err <= t)) for t in thresholds}

    out = {
        "model": name,
        "n": int(err.shape[0]),
        "rmse_2d_m": float(rmse_2d),
        "mae_2d_m": float(mae_2d),
        "median_err_m": float(r50),
        "p68_err_m": float(r68),
        "p90_err_m": float(r90),
        "p95_err_m": float(r95),
        "p99_err_m": float(r99),
        "max_err_m": float(np.max(err)),
        "rmse_x_m": float(rmse_x),
        "rmse_y_m": float(rmse_y),
        "mae_x_m": float(mae_x),
        "mae_y_m": float(mae_y),
        "bias_x_m": float(bias_x),
        "bias_y_m": float(bias_y),
        "errors_radial_m": err,
    }

    out.update(cov)
    return out
