"""Plotting utilities (no training)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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


def plot_topk_corr_heatmap(
    df: pd.DataFrame,
    *,
    target_cols: Sequence[str] = ("label_X", "label_Y"),
    candidate_cols: Sequence[str] | None = None,
    topk_k: int | None = None,
    exclude_cols: Sequence[str] = ("t_ms", "session_id", "segment_id"),
    min_non_null: int = 50,
    title: str = "Top correlated features vs labels",
    save_path: Path | None = None,
) -> Dict[str, object] | None:
    """Plot correlation heatmap for candidate features vs target labels.

    If topk_k is None or <= 0, all candidate features are included.
    """

    if df is None or len(df) == 0:
        return None

    targets = [c for c in target_cols if c in df.columns]
    if not targets:
        return None

    if candidate_cols is None:
        cand = [c for c in df.columns if c not in set(targets)]
    else:
        cand = [c for c in candidate_cols if c in df.columns]

    exclude = set(exclude_cols)
    cand = [c for c in cand if c not in exclude]

    numeric_cols = []
    for c in cand:
        s = pd.to_numeric(df[c], errors="coerce")
        if int(s.notna().sum()) >= int(min_non_null):
            numeric_cols.append(c)

    if not numeric_cols:
        return None

    subset = df[targets + numeric_cols].copy()
    for c in subset.columns:
        subset[c] = pd.to_numeric(subset[c], errors="coerce")

    corr = subset.corr()
    scores: Dict[str, float] = {}
    for c in numeric_cols:
        vals = [abs(corr.loc[c, t]) for t in targets if t in corr.index]
        if vals and np.isfinite(vals).any():
            scores[c] = float(np.nanmean(vals))

    if not scores:
        return None

    ordered = sorted(scores, key=scores.get, reverse=True)
    if topk_k is not None and int(topk_k) > 0:
        ordered = ordered[: int(topk_k)]

    cols = list(targets) + ordered
    plot_corr_heatmap(subset, cols, title=title, save_path=save_path)
    return {"cols": cols, "scores": scores, "selected": ordered}


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


def plot_axis_error_boxplot_subset(
    y_true_eval: np.ndarray,
    pred_map: Dict[str, np.ndarray],
    *,
    model_order: Sequence[str],
    title: str,
    save_path: Path | None = None,
):
    preds_for_axis = {k: v for k, v in pred_map.items() if k in model_order and v is not None}
    return plot_axis_error_boxplot(
        y_true_eval,
        preds_for_axis,
        title=title,
        save_path=save_path,
    )


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


def plot_cumulative_mean_error(
    y_true: np.ndarray,
    preds_by_model: Dict[str, np.ndarray],
    *,
    model_order: Sequence[str],
    title: str = "Cumulative mean radial error",
    save_path: Path | None = None,
) -> None:
    """Plot cumulative mean radial error curves for selected models."""

    y_true = np.asarray(y_true, dtype=float)
    plt.figure(figsize=(9, 4))

    for name in model_order:
        pred = preds_by_model.get(name)
        if pred is None:
            continue
        yp = np.asarray(pred, dtype=float)
        n = min(len(y_true), len(yp))
        if n == 0:
            continue
        yt = y_true[:n]
        yp = yp[:n]
        mask = np.isfinite(yt).all(axis=1) & np.isfinite(yp).all(axis=1)
        if int(mask.sum()) == 0:
            continue
        err = np.linalg.norm(yp[mask] - yt[mask], axis=1)
        curve = np.cumsum(err) / (np.arange(len(err)) + 1)
        plt.plot(curve, label=name)

    plt.title(title)
    plt.xlabel("ordered target index")
    plt.ylabel("cumulative mean error (m)")
    plt.grid(True)
    plt.legend(ncol=2)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=160)
    plt.show(block=False)


def plot_error_by_dt_bin(
    df_ref: pd.DataFrame,
    y_true: np.ndarray,
    preds_by_model: Dict[str, np.ndarray],
    *,
    model_order: Sequence[str],
    time_col: str = "t_ms",
    bins: Tuple[float, ...] = (0.0, 100.0, 200.0, 400.0, 800.0, 1500.0, np.inf),
    labels: Tuple[str, ...] = ("<=100", "100-200", "200-400", "400-800", "800-1500", ">1500"),
    title: str = "Median error by dt bin",
    save_path: Path | None = None,
) -> pd.DataFrame:
    """Plot median radial error by local dt bins and return the underlying table."""

    if len(bins) - 1 != len(labels):
        raise ValueError("labels must have len(bins)-1 elements")

    df = df_ref.copy()
    if "session_id" in df.columns:
        df["dt_ms"] = df.groupby("session_id")[time_col].diff() if time_col in df.columns else np.nan
    else:
        df["dt_ms"] = df[time_col].diff() if time_col in df.columns else np.nan
    df["dt_bin"] = pd.cut(df["dt_ms"], bins=bins, labels=labels)

    y_true = np.asarray(y_true, dtype=float)
    rows = []
    for name in model_order:
        pred = preds_by_model.get(name)
        if pred is None:
            continue
        yp = np.asarray(pred, dtype=float)
        n = min(len(yp), len(y_true), len(df))
        if n == 0:
            continue
        yt = y_true[:n]
        yp = yp[:n]
        mask = np.isfinite(yt).all(axis=1) & np.isfinite(yp).all(axis=1)
        if int(mask.sum()) < 10:
            continue
        tmp = pd.DataFrame(
            {
                "dt_bin": df["dt_bin"].iloc[:n].to_numpy(),
                "err": np.linalg.norm(yp - yt, axis=1),
                "ok": mask,
            }
        )
        tmp = tmp[tmp["ok"]]
        if tmp.empty:
            continue
        rows.append(tmp.groupby("dt_bin", observed=False)["err"].median().rename(name))

    if not rows:
        return pd.DataFrame()

    dt_err = pd.concat(rows, axis=1)
    dt_err.plot(figsize=(9, 4), marker="o")
    plt.title(title)
    plt.xlabel("dt bin (ms)")
    plt.ylabel("median radial error (m)")
    plt.grid(True)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=160)
    plt.show(block=False)
    return dt_err


def plot_error_by_dt_bin_report(
    df_eval: pd.DataFrame,
    y_true_eval: np.ndarray,
    pred_map: Dict[str, np.ndarray],
    *,
    model_order: Sequence[str],
    time_col: str,
    save_path: Path | None = None,
):
    from IPython.display import display

    dt_err = plot_error_by_dt_bin(
        df_eval,
        y_true_eval,
        pred_map,
        model_order=model_order,
        time_col=time_col,
        save_path=save_path,
    )
    if len(dt_err):
        display(dt_err)
    return dt_err


def summarize_segment_errors(
    df_ref: pd.DataFrame,
    y_true: np.ndarray,
    preds_by_model: Dict[str, np.ndarray],
    *,
    candidate_models: Sequence[str] = ("LSTM_FE", "GRU_FE", "LSTM_FE_KF", "GRU_FE_KF"),
    best_model: str | None = None,
) -> Tuple[str | None, pd.DataFrame]:
    """Return best sequential model and segment-level error summary."""

    available = [m for m in candidate_models if m in preds_by_model]
    if not available:
        return None, pd.DataFrame()

    y_true = np.asarray(y_true, dtype=float)
    if best_model is None:
        best_model = None
        best_median = np.inf
        for name in available:
            yp = np.asarray(preds_by_model[name], dtype=float)
            n = min(len(yp), len(y_true))
            if n == 0:
                continue
            mask = np.isfinite(y_true[:n]).all(axis=1) & np.isfinite(yp[:n]).all(axis=1)
            if int(mask.sum()) < 10:
                continue
            med = float(np.median(np.linalg.norm(yp[:n][mask] - y_true[:n][mask], axis=1)))
            if med < best_median:
                best_median = med
                best_model = name

    if best_model is None or best_model not in preds_by_model:
        return None, pd.DataFrame()

    needed = [c for c in ["session_id", "segment_id", "t_ms"] if c in df_ref.columns]
    if "session_id" not in needed or "segment_id" not in needed:
        return best_model, pd.DataFrame()

    yp = np.asarray(preds_by_model[best_model], dtype=float)
    n = min(len(yp), len(y_true), len(df_ref))
    if n == 0:
        return best_model, pd.DataFrame()

    err = np.linalg.norm(yp[:n] - y_true[:n], axis=1)
    seg_eval = df_ref.iloc[:n][needed].copy()
    seg_eval["err"] = err
    seg_eval = seg_eval[np.isfinite(seg_eval["err"])].copy()
    if seg_eval.empty:
        return best_model, pd.DataFrame()

    seg_stats = (
        seg_eval.groupby(["session_id", "segment_id"])
        .agg(
            n_points=("err", "size"),
            median_err_m=("err", "median"),
            p90_err_m=("err", lambda s: float(np.nanquantile(np.asarray(s, dtype=float), 0.9))),
        )
        .sort_values("median_err_m")
    )
    return best_model, seg_stats


def display_segment_error_summary(
    df_eval: pd.DataFrame,
    y_true_eval: np.ndarray,
    pred_map: Dict[str, np.ndarray],
):
    from IPython.display import display

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        out = df.replace([np.inf, -np.inf], np.nan).copy()
        out = out.dropna(axis=1, how="all")
        return out.fillna("n/a")

    best_seq_name, seg_stats = summarize_segment_errors(
        df_eval,
        y_true_eval,
        pred_map,
    )

    if best_seq_name is None or len(seg_stats) == 0:
        print("No sequential predictions available.")
    else:
        print("Best sequential model:", best_seq_name)
        print("Best 10 segments")
        display(_clean(seg_stats.head(10)))
        print("Worst 10 segments")
        display(_clean(seg_stats.tail(10)))

    return best_seq_name, seg_stats


def plot_segment_prediction_comparison(
    df_ref: pd.DataFrame,
    y_true: np.ndarray,
    preds_by_model: Dict[str, np.ndarray],
    *,
    session_id: str,
    segment_id: int,
    model_order: Sequence[str],
    title_prefix: str,
    save_path: Path | None = None,
    show_arrows: bool = True,
    max_arrows: int = 12,
) -> bool:
    """Plot true/pred trajectories for one session/segment selection."""

    if "session_id" not in df_ref.columns or "segment_id" not in df_ref.columns:
        return False
    if "t_ms" not in df_ref.columns:
        return False

    mask = (df_ref["session_id"] == session_id) & (df_ref["segment_id"] == segment_id)
    idx = np.where(mask.to_numpy())[0]
    if len(idx) < 2:
        return False
    idx_ord = idx[np.argsort(df_ref.loc[mask, "t_ms"].to_numpy())]

    y_true = np.asarray(y_true, dtype=float)
    n_ref = len(y_true)
    idx_ord = idx_ord[idx_ord < n_ref]
    if len(idx_ord) < 2:
        return False

    fig = plt.figure(figsize=(7, 6))
    ax = plt.gca()

    def _add_arrows(xy: np.ndarray, *, color: str, alpha: float = 0.9) -> None:
        if not show_arrows:
            return
        if xy.shape[0] < 2:
            return
        n = int(xy.shape[0])
        step = max(1, int(n // max(1, max_arrows)))
        for i in range(0, n - 1, step):
            p0 = xy[i]
            p1 = xy[i + 1]
            if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
                continue
            ax.annotate(
                "",
                xy=(p1[0], p1[1]),
                xytext=(p0[0], p0[1]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0, alpha=alpha),
            )

    true_line = ax.plot(y_true[idx_ord, 0], y_true[idx_ord, 1], label="True", linewidth=2, color="black")[0]
    _add_arrows(y_true[idx_ord], color=true_line.get_color(), alpha=0.9)

    lines = 0
    for name in model_order:
        pred = preds_by_model.get(name)
        if pred is None:
            continue
        yp = np.asarray(pred, dtype=float)
        valid_idx = idx_ord[idx_ord < len(yp)]
        if len(valid_idx) < 2:
            continue
        seg_pred = yp[valid_idx]
        seg_true = y_true[valid_idx]
        ok = np.isfinite(seg_pred).all(axis=1) & np.isfinite(seg_true).all(axis=1)
        if int(ok.sum()) < 2:
            continue
        line = ax.plot(seg_pred[ok, 0], seg_pred[ok, 1], label=name)[0]
        _add_arrows(seg_pred[ok], color=line.get_color(), alpha=0.7)
        lines += 1

    if lines == 0:
        plt.close()
        return False

    plt.title(f"{title_prefix} - session={session_id}, segment={segment_id}")
    plt.axis("equal")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=160)
    plt.show(block=False)
    return True


def summarize_cross_device_tables(
    cross_df: pd.DataFrame,
    *,
    seq_order: Sequence[str] = ("LSTM_FE", "LSTM_FE_KF", "GRU_FE", "GRU_FE_KF"),
) -> Dict[str, pd.DataFrame]:
    """Prepare family-specific cross-device tables and pivots."""

    out = {
        "cross_seq": pd.DataFrame(),
        "lstm_table": pd.DataFrame(),
        "gru_table": pd.DataFrame(),
        "lstm_pivot": pd.DataFrame(),
        "gru_pivot": pd.DataFrame(),
    }
    if cross_df is None or len(cross_df) == 0:
        return out

    available_models = [m for m in seq_order if m in cross_df["model"].unique()]
    if not available_models:
        return out

    cross_seq = cross_df[cross_df["model"].isin(available_models)].copy()
    base_cols = ["pair", "model", "median_err_m", "p90_err_m", "rmse_2d_m", "n", "test_coverage"]

    lstm_df = cross_seq[cross_seq["model"].str.startswith("LSTM")][base_cols]
    gru_df = cross_seq[cross_seq["model"].str.startswith("GRU")][base_cols]
    lstm_df = lstm_df.sort_values(["pair", "median_err_m"]).reset_index(drop=True)
    gru_df = gru_df.sort_values(["pair", "median_err_m"]).reset_index(drop=True)

    out["cross_seq"] = cross_seq
    out["lstm_table"] = lstm_df
    out["gru_table"] = gru_df
    if len(lstm_df):
        out["lstm_pivot"] = lstm_df.pivot_table(index="pair", columns="model", values="median_err_m", aggfunc="min")
    if len(gru_df):
        out["gru_pivot"] = gru_df.pivot_table(index="pair", columns="model", values="median_err_m", aggfunc="min")
    return out


def plot_best_model_full_trajectory(
    df_ref: pd.DataFrame,
    y_true: np.ndarray,
    preds_by_model: Dict[str, np.ndarray],
    *,
    metrics_df: pd.DataFrame | None = None,
    model_candidates: Sequence[str] = ("LSTM_FE", "GRU_FE", "LSTM_FE_KF", "GRU_FE_KF"),
    time_col: str = "t_ms",
    title: str = "Best model on full trajectory",
    save_path: Path | None = None,
) -> Dict[str, object] | None:
    """Plot best sequential model on the longest available session trajectory."""

    if df_ref is None or len(df_ref) == 0:
        return None
    if "session_id" not in df_ref.columns:
        return None

    y_true = np.asarray(y_true, dtype=float)
    best_model = None

    if metrics_df is not None and len(metrics_df):
        cand = metrics_df[metrics_df["model"].isin(list(model_candidates))].sort_values("median_err_m")
        if len(cand):
            best_model = str(cand.iloc[0]["model"])

    if best_model is None:
        best_median = np.inf
        for name in model_candidates:
            if name not in preds_by_model:
                continue
            yp = np.asarray(preds_by_model[name], dtype=float)
            n = min(len(yp), len(y_true))
            if n == 0:
                continue
            mask = np.isfinite(y_true[:n]).all(axis=1) & np.isfinite(yp[:n]).all(axis=1)
            if int(mask.sum()) < 10:
                continue
            med = float(np.median(np.linalg.norm(yp[:n][mask] - y_true[:n][mask], axis=1)))
            if med < best_median:
                best_median = med
                best_model = name

    if best_model is None or best_model not in preds_by_model:
        return None

    yp = np.asarray(preds_by_model[best_model], dtype=float)
    n = min(len(df_ref), len(y_true), len(yp))
    if n < 2:
        return None

    df = df_ref.iloc[:n].copy()
    yt = y_true[:n]
    yp = yp[:n]

    valid = np.isfinite(yt).all(axis=1) & np.isfinite(yp).all(axis=1)
    if int(valid.sum()) < 2:
        return None

    df["__valid"] = valid
    counts = df.groupby("session_id")["__valid"].sum().sort_values(ascending=False)
    if counts.empty:
        return None
    session_id = counts.index[0]

    df_s = df[df["session_id"] == session_id].copy()
    if time_col in df_s.columns:
        df_s = df_s.sort_values(time_col, kind="mergesort")
    idx = df_s.index.to_numpy(dtype=int)

    yt_s = yt[idx]
    yp_s = yp[idx]
    true_mask = np.isfinite(yt_s).all(axis=1)
    pred_mask = np.isfinite(yp_s).all(axis=1)
    if int(true_mask.sum()) < 2 or int(pred_mask.sum()) < 2:
        return None

    plt.figure(figsize=(7.5, 6.5))
    plt.plot(yt_s[true_mask, 0], yt_s[true_mask, 1], label="True", linewidth=2.0, color="black")
    plt.plot(yp_s[pred_mask, 0], yp_s[pred_mask, 1], label=best_model, linewidth=1.8, color="tab:orange")
    plt.title(f"{title} | {best_model} | session={session_id}")
    plt.axis("equal")
    plt.grid(True, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=160)
    plt.show(block=False)

    return {
        "model": best_model,
        "session_id": session_id,
        "n_points": int(pred_mask.sum()),
    }


def plot_cross_device_heatmaps(
    cross_df: pd.DataFrame,
    *,
    seq_order: Sequence[str] = ("LSTM_FE", "LSTM_FE_KF", "GRU_FE", "GRU_FE_KF"),
    save_path: Path | None = None,
    title: str = "Cross-device median error heatmaps",
) -> List[str]:
    """Plot one source->target heatmap per sequential model."""

    if cross_df is None or len(cross_df) == 0:
        return []
    available_models = [m for m in seq_order if m in cross_df["model"].unique()]
    if not available_models:
        return []

    cross_seq = cross_df[cross_df["model"].isin(available_models)].copy()
    n_models = len(available_models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), squeeze=False, constrained_layout=True)
    last_im = None
    plotted = []

    for ax, model_name in zip(axes[0], available_models):
        mat = cross_seq[cross_seq["model"] == model_name].pivot_table(
            index="source", columns="target", values="median_err_m", aggfunc="median"
        )
        if mat.empty:
            ax.set_axis_off()
            continue

        values = mat.values.astype(float)
        last_im = ax.imshow(values, cmap="viridis")
        ax.set_title(f"{model_name}\nmedian error (m)")
        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_xticklabels(mat.columns, rotation=30, ha="right")
        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_yticklabels(mat.index)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white", fontsize=9)
        plotted.append(model_name)

    if last_im is not None:
        fig.colorbar(last_im, ax=axes[0].tolist(), shrink=0.8, label="median error (m)")
    fig.suptitle(title)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=160)
    plt.show(block=False)
    return plotted


def display_cross_device_tables_and_heatmaps(cross_df: pd.DataFrame, *, save_path: Path | None = None):
    from IPython.display import display

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        out = df.replace([np.inf, -np.inf], np.nan).copy()
        out = out.dropna(axis=1, how="all")
        return out.fillna("n/a")

    if cross_df is None or len(cross_df) == 0:
        print("No cross-device data are available for visualization.")
        return {"tables": {}, "plotted": False}

    tables = summarize_cross_device_tables(cross_df)

    if len(tables.get("lstm_table", pd.DataFrame())):
        print("LSTM table (within-family comparison)")
        display(_clean(tables["lstm_table"]))
        print("Pivot median_err_m (LSTM)")
        display(_clean(tables["lstm_pivot"]))
    else:
        print("No LSTM variant is available.")

    if len(tables.get("gru_table", pd.DataFrame())):
        print("GRU table (within-family comparison)")
        display(_clean(tables["gru_table"]))
        print("Pivot median_err_m (GRU)")
        display(_clean(tables["gru_pivot"]))
    else:
        print("No GRU variant is available.")

    plotted = plot_cross_device_heatmaps(
        cross_df,
        save_path=save_path,
    )
    if not plotted:
        print("No sequential model is available for heatmaps.")

    return {"tables": tables, "plotted": bool(plotted)}


def plot_best_worst_segments(
    df_eval: pd.DataFrame,
    y_true_eval: np.ndarray,
    pred_map: Dict[str, np.ndarray],
    seg_stats: pd.DataFrame,
    *,
    save_dir: Path,
    run_id: str,
    model_order: Sequence[str] | None = None,
):
    if seg_stats is None or not isinstance(seg_stats, pd.DataFrame) or len(seg_stats) == 0:
        print("No segment statistics available for best/worst plots.")
        return []

    if model_order is None:
        model_order = ["baseline_constant_velocity_rollout", "LSTM_FE", "GRU_FE", "LSTM_FE_KF", "GRU_FE_KF"]

    sid_best, seg_best = seg_stats.index[0]
    sid_worst, seg_worst = seg_stats.index[-1]

    plot_segment_prediction_comparison(
        df_eval,
        y_true_eval,
        pred_map,
        session_id=sid_best,
        segment_id=seg_best,
        model_order=model_order,
        title_prefix="best segment",
        save_path=save_dir / f"traj_best_segment__{run_id}.png",
    )

    plot_segment_prediction_comparison(
        df_eval,
        y_true_eval,
        pred_map,
        session_id=sid_worst,
        segment_id=seg_worst,
        model_order=model_order,
        title_prefix="worst segment",
        save_path=save_dir / f"traj_worst_segment__{run_id}.png",
    )

    return [(sid_best, seg_best), (sid_worst, seg_worst)]


def plot_cross_device_best_model_cdf(
    cross_results: List[Dict],
    cross_df: pd.DataFrame,
    *,
    desired_models: Sequence[str] = ("LSTM_FE", "GRU_FE", "LSTM_FE_KF", "GRU_FE_KF"),
    save_path: Path | None = None,
) -> str | None:
    """Plot CDFs by source->target pair for the best cross-device model."""

    if not cross_results or cross_df is None or len(cross_df) == 0:
        return None

    models_in_preds = set(d.get("model") for d in cross_results)
    models_in_metrics = set(cross_df["model"].unique())
    usable_models = [m for m in desired_models if m in models_in_preds and m in models_in_metrics]
    if not usable_models:
        return None

    best_model = (
        cross_df[cross_df["model"].isin(usable_models)]
        .sort_values("median_err_m")
        .iloc[0]["model"]
    )

    plt.figure(figsize=(8, 5))
    lines_plotted = 0
    for item in cross_results:
        if item.get("model") != best_model:
            continue
        y_true = np.asarray(item.get("y_true"), dtype=float)
        y_pred = np.asarray(item.get("y_pred"), dtype=float)
        n = min(len(y_true), len(y_pred))
        if n < 10:
            continue
        yt = y_true[:n]
        yp = y_pred[:n]
        mask = np.isfinite(yt).all(axis=1) & np.isfinite(yp).all(axis=1)
        if int(mask.sum()) < 10:
            continue
        err = np.sort(np.linalg.norm(yp[mask] - yt[mask], axis=1))
        cdf = np.arange(1, len(err) + 1) / len(err)
        plt.plot(err, cdf, label=f"{item.get('pair', 'pair')} (n={len(err)})")
        lines_plotted += 1

    if lines_plotted == 0:
        plt.close()
        return None

    plt.title(f"Cross-device CDF - {best_model}")
    plt.xlabel("radial error (m)")
    plt.ylabel("CDF")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=160)
    plt.show(block=False)
    return str(best_model)


def plot_cross_device_trajectories(
    cross_results: List[Dict],
    cross_df: pd.DataFrame,
    *,
    save_dir: Path | None = None,
    run_id: str | None = None,
) -> List[str]:
    """Plot pairwise trajectory overlays with one subplot per model family."""

    if not cross_results or cross_df is None or len(cross_df) == 0:
        return []

    families = {
        "LSTM": ["LSTM_FE", "LSTM_FE_KF"],
        "GRU": ["GRU_FE", "GRU_FE_KF"],
    }
    style_map = {
        "LSTM_FE": {"color": "tab:red", "linestyle": "-", "linewidth": 1.8},
        "LSTM_FE_KF": {"color": "tab:purple", "linestyle": "--", "linewidth": 2.0},
        "GRU_FE": {"color": "tab:blue", "linestyle": "-", "linewidth": 1.8},
        "GRU_FE_KF": {"color": "tab:gray", "linestyle": "--", "linewidth": 2.0},
    }

    models_in_preds = set(d.get("model") for d in cross_results)
    pair_list = sorted(set((d.get("source"), d.get("target")) for d in cross_results if d.get("source") and d.get("target")))
    plotted_pairs: List[str] = []

    def _select_indices(pack: Dict) -> np.ndarray:
        y_true = np.asarray(pack.get("y_true"), dtype=float)
        n = len(y_true)
        idx = np.arange(n)
        sid = pack.get("session_id")
        seg = pack.get("segment_id")
        tms = pack.get("t_ms")
        if sid is not None and seg is not None and len(np.asarray(sid)) == n and len(np.asarray(seg)) == n:
            meta = pd.DataFrame(
                {
                    "idx": np.arange(n),
                    "sid": np.asarray(sid)[:n],
                    "seg": np.asarray(seg)[:n],
                }
            )
            if tms is not None and len(np.asarray(tms)) == n:
                meta["t"] = np.asarray(tms)[:n]
            largest = meta.groupby(["sid", "seg"]).size().sort_values(ascending=False)
            if len(largest):
                sid0, seg0 = largest.index[0]
                sub = meta[(meta["sid"] == sid0) & (meta["seg"] == seg0)].copy()
                sub = sub.sort_values("t") if "t" in sub.columns else sub.sort_values("idx")
                idx = sub["idx"].to_numpy(dtype=int)
        return idx

    for src, tgt in pair_list:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), squeeze=False)
        axes = axes[0]
        any_plot = False

        for ax, (fam_name, fam_models_all) in zip(axes, families.items()):
            fam_models = [m for m in fam_models_all if m in models_in_preds]
            pair_preds = {
                d["model"]: d
                for d in cross_results
                if d.get("source") == src and d.get("target") == tgt and d.get("model") in fam_models
            }
            if not pair_preds:
                ax.set_axis_off()
                continue
            pair_df = cross_df[
                (cross_df["source"] == src)
                & (cross_df["target"] == tgt)
                & (cross_df["model"].isin(pair_preds.keys()))
            ]
            if pair_df.empty:
                ax.set_axis_off()
                continue
            ref_model = pair_df.sort_values("median_err_m").iloc[0]["model"]
            ref_pack = pair_preds.get(ref_model)
            if ref_pack is None:
                ax.set_axis_off()
                continue
            indices = _select_indices(ref_pack)
            y_true_ref = np.asarray(ref_pack["y_true"], dtype=float)
            true_xy = y_true_ref[indices]
            if len(true_xy) < 2:
                ax.set_axis_off()
                continue

            ax.plot(true_xy[:, 0], true_xy[:, 1], color="black", linewidth=2.4, label="True", zorder=3)
            for model_name in fam_models_all:
                pack = pair_preds.get(model_name)
                if pack is None:
                    continue
                y_pred = np.asarray(pack.get("y_pred"), dtype=float)
                if len(y_pred) == 0:
                    continue
                valid_idx = indices[indices < len(y_pred)]
                if len(valid_idx) < 2:
                    continue
                pred_xy = y_pred[valid_idx]
                mask = np.isfinite(pred_xy).all(axis=1)
                pred_xy = pred_xy[mask]
                if len(pred_xy) < 2:
                    continue
                st = style_map.get(model_name, {})
                ax.plot(pred_xy[:, 0], pred_xy[:, 1], alpha=0.95, label=model_name, zorder=2, **st)

            ax.set_title(f"{fam_name} | ref: {ref_model}, points={len(indices)}")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.grid(True, alpha=0.35)
            ax.set_aspect("equal", adjustable="box")
            ax.legend(loc="best", frameon=True)
            any_plot = True

        if not any_plot:
            plt.close(fig)
            continue
        fig.suptitle(f"Cross-device trajectories - {src} -> {tgt}", y=1.03)
        plt.tight_layout()
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"__{run_id}" if run_id else ""
            out = save_dir / f"cross_device_trajectories_{src}_to_{tgt}{suffix}.png"
            plt.savefig(out, dpi=160)
        plt.show(block=False)
        plotted_pairs.append(f"{src}->{tgt}")

    return plotted_pairs


def plot_cross_device_cdf_and_trajectories(
    cross_results: List[Dict],
    cross_df: pd.DataFrame,
    *,
    save_dir: Path,
    run_id: str,
):
    best_model = plot_cross_device_best_model_cdf(
        cross_results,
        cross_df,
        save_path=save_dir / f"cross_device_cdf_best_model__{run_id}.png",
    )
    if best_model is None:
        print("Cross-device CDF is not available.")

    pairs = plot_cross_device_trajectories(
        cross_results,
        cross_df,
        save_dir=save_dir,
        run_id=run_id,
    )
    if len(pairs) == 0:
        print("Cross-device trajectories are not available.")

    return best_model, pairs
