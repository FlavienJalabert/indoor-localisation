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
    min_points: int = 10,
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
        if len(err) < min_points:
            continue
        cdf = np.arange(1, len(err) + 1) / len(err)
        label = f"{row.get('model', 'model')} (n={len(err)})"
        plt.plot(err, cdf, label=label)

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


def plot_axis_error_boxplot(
    y_true: np.ndarray,
    preds: Dict[str, np.ndarray],
    *,
    title: str,
    save_path: Path | None = None,
) -> None:
    """Plot absolute error distributions per axis for each model."""

    y_true = np.asarray(y_true, dtype=float)
    data_x = []
    data_y = []
    labels = []
    for name, y_pred in preds.items():
        yp = np.asarray(y_pred, dtype=float)
        n = min(len(y_true), len(yp))
        if n == 0:
            continue
        yt = y_true[:n]
        yp = yp[:n]
        mask = np.isfinite(yt).all(axis=1) & np.isfinite(yp).all(axis=1)
        if mask.sum() < 10:
            continue
        ex = np.abs(yp[mask, 0] - yt[mask, 0])
        ey = np.abs(yp[mask, 1] - yt[mask, 1])
        data_x.append(ex)
        data_y.append(ey)
        labels.append(name)

    if not labels:
        return

    plt.figure(figsize=(max(8, len(labels) * 1.6), 4.5))
    positions = np.arange(len(labels))
    dx = 0.18
    bp_x = plt.boxplot(data_x, positions=positions - dx, widths=0.3, patch_artist=True)
    bp_y = plt.boxplot(data_y, positions=positions + dx, widths=0.3, patch_artist=True)
    for box in bp_x["boxes"]:
        box.set_facecolor("#4C72B0")
        box.set_alpha(0.45)
    for box in bp_y["boxes"]:
        box.set_facecolor("#DD8452")
        box.set_alpha(0.45)
    plt.xticks(positions, labels, rotation=20, ha="right")
    plt.ylabel("|error| (m)")
    plt.title(title)
    plt.legend([bp_x["boxes"][0], bp_y["boxes"][0]], ["|err_x|", "|err_y|"], loc="upper right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show(block=False)


def plot_cross_device_result(res: Dict, *, save_dir: Path | None = None) -> None:
    """Plot cross-device trajectory and error curve from a result dict."""

    if "y_true" not in res or "y_pred" not in res:
        return

    y_true = np.asarray(res["y_true"], dtype=float)
    y_pred = np.asarray(res["y_pred"], dtype=float)
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    t_ms = res.get("t_ms")
    session_id = res.get("session_id")
    segment_id = res.get("segment_id")
    anchor_x = res.get("anchor_X")
    anchor_y = res.get("anchor_Y")
    if anchor_x is not None and anchor_y is not None:
        ax = np.asarray(anchor_x, dtype=float)[:n]
        ay = np.asarray(anchor_y, dtype=float)[:n]
        m_anchor = np.isfinite(ax) & np.isfinite(ay)
        if m_anchor.sum() >= 4:
            y_true = y_true.copy()
            y_true[m_anchor, 0] = ax[m_anchor]
            y_true[m_anchor, 1] = ay[m_anchor]

    meta = pd.DataFrame({"idx": np.arange(n)})
    if t_ms is not None:
        meta["t_ms"] = np.asarray(t_ms)[:n]
    if session_id is not None:
        meta["session_id"] = np.asarray(session_id)[:n]
    if segment_id is not None:
        meta["segment_id"] = np.asarray(segment_id)[:n]

    sort_cols = [c for c in ("session_id", "segment_id", "t_ms", "idx") if c in meta.columns]
    if sort_cols:
        meta = meta.sort_values(sort_cols, kind="mergesort")
    order = meta["idx"].to_numpy(dtype=int)
    y_true_ord = y_true[order]
    y_pred_ord = y_pred[order]
    err = np.linalg.norm(y_pred_ord - y_true_ord, axis=1)

    group_cols = [c for c in ("session_id", "segment_id") if c in meta.columns]
    grouped = [(None, meta)] if not group_cols else list(meta.groupby(group_cols, sort=False))
    if grouped:
        # Plot only the largest coherent trajectory to avoid cross-session stitching artefacts.
        grouped = [max(grouped, key=lambda kv: len(kv[1]))]

    def _compress_path(arr: np.ndarray) -> np.ndarray:
        if len(arr) <= 1:
            return arr
        keep = np.ones(len(arr), dtype=bool)
        d = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        keep[1:] = d > 1e-9
        return arr[keep]

    plt.figure(figsize=(6, 6))
    first = True
    for _, g in grouped:
        idx = g["idx"].to_numpy(dtype=int)
        y_t = _compress_path(y_true[idx])
        y_p = _compress_path(y_pred[idx])
        if len(y_t):
            plt.plot(y_t[:, 0], y_t[:, 1], color="tab:blue", label="True" if first else None)
        if len(y_p):
            plt.plot(y_p[:, 0], y_p[:, 1], color="tab:orange", label="Pred" if first else None)
        first = False
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
