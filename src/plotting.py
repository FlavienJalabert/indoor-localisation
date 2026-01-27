"""Plotting utilities (no training)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None


def plot_trajectory(
    df: pd.DataFrame,
    *,
    x: str = "label_X",
    y: str = "label_Y",
    color=None,
    title: str = "",
    save_path: Path | None = None,
) -> None:
    """Scatter trajectory plot."""

    plt.figure(figsize=(7, 7))
    plt.scatter(df[x], df[y], s=5, alpha=0.5, c=color)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.axis("equal")
    plt.grid(True)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show(block=False)


def plot_corr_heatmap(
    df: pd.DataFrame, cols: List[str], *, title: str = "", save_path: Path | None = None
) -> None:
    """Correlation heatmap for selected columns."""

    corr = df[cols].corr()
    plt.figure(figsize=(10, 6))
    if sns is not None:
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    else:
        plt.imshow(corr, cmap="coolwarm")
        plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show(block=False)


def plot_error_cdf(
    results_df: pd.DataFrame,
    *,
    title: str,
    max_m: float | None,
    save_path: Path | None = None,
) -> None:
    """Plot CDFs of radial errors for each model (expects errors_radial_m column)."""

    plt.figure(figsize=(7, 5))
    for _, row in results_df.iterrows():
        if "errors_radial_m" not in row:
            continue
        err = np.asarray(row["errors_radial_m"], dtype=float)
        err = err[~np.isnan(err)]
        err = np.sort(err)
        if len(err) == 0:
            continue
        cdf = np.arange(1, len(err) + 1) / len(err)
        plt.plot(err, cdf, label=row.get("model", "model"))

    if max_m is not None:
        plt.xlim([0, max_m])
    plt.xlabel("Radial error (m)")
    plt.ylabel("CDF")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show(block=False)


def plot_cross_device_result(res: Dict, *, save_dir: Path | None = None) -> None:
    """Plot cross-device trajectory and error curve from a result dict."""

    if "y_true" not in res or "y_pred" not in res:
        return

    y_true = np.asarray(res["y_true"])
    y_pred = np.asarray(res["y_pred"])
    err = np.linalg.norm(y_pred - y_true, axis=1)

    plt.figure(figsize=(6, 6))
    plt.plot(y_true[:, 0], y_true[:, 1], label="True")
    plt.plot(y_pred[:, 0], y_pred[:, 1], label="Pred")
    plt.legend()
    plt.title(res.get("name", "cross-device"))
    plt.axis("equal")
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"traj_{res.get('name','run')}.png", dpi=150)
    plt.show(block=False)

    plt.figure(figsize=(7, 4))
    plt.plot(err)
    plt.title("Error over time")
    plt.xlabel("Index")
    plt.ylabel("Error (m)")
    if save_dir is not None:
        plt.savefig(save_dir / f"err_{res.get('name','run')}.png", dpi=150)
    plt.show(block=False)


def plot_cross_device_comparison(results: List[Dict], *, save_path: Path | None = None) -> None:
    """Overlay cross-device error CDFs for multiple results."""

    plt.figure(figsize=(7, 5))
    for res in results:
        if "y_true" not in res or "y_pred" not in res:
            continue
        y_true = np.asarray(res["y_true"])
        y_pred = np.asarray(res["y_pred"])
        err = np.linalg.norm(y_pred - y_true, axis=1)
        err = np.sort(err)
        cdf = np.arange(1, len(err) + 1) / len(err)
        plt.plot(err, cdf, label=res.get("name", "run"))

    plt.xlabel("Radial error (m)")
    plt.ylabel("CDF")
    plt.title("Cross-device comparison")
    plt.grid(True)
    plt.legend()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show(block=False)
