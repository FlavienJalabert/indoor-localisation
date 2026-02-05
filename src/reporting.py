"""Notebook reporting helpers to keep notebooks concise and figure-driven."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from config import Config, set_global_seed
import evaluation


CONCLUSION_TEXT = """Overall, the project delivers a coherent temporal pipeline for indoor localization from WiFi+IMU streams, with a leakage-safe evaluation protocol and explicit cross-device checks. The experiments show that sequence models can recover within-device trajectories under dense labels, but cross-device transfer remains the dominant limitation.

**Does it answer the base problem?**  
Within a single device and under the dense protocol, yes: LSTM_FE_KF reaches a median error of 4.391 m (p90 7.152 m) and is essentially tied with LSTM_FE (4.438 m). Coverage is 0.976 (1355/1388 windows) with a 160 ms densify step (test densify factor 1.47).

**What is solid and informative**
- Correlation-based feature selection is critical: turning top-k off adds +2.331 m to the median error, and k=10 is even worse (+3.321 m).
- Window-length perturbations degrade results (short window +2.232 m, long window +1.451 m), which suggests the base temporal context is close to optimal for this dataset.
- Strict-time checks did not change support or metrics in this run, indicating that the current densified setup does not stress time consistency.

**What limits the system**
- Dense labels exhibit staircase behavior, which makes dense-aligned evaluation optimistic relative to realistic, sparse label availability.
- Cross-device generalization is weak and asymmetric: esp32->samsung reaches a 14.723 m median (p90 63.608 m) while samsung->esp32 reaches 32.692 m (p90 91.278 m), with heavy tails in both directions.
- Cross-split robustness is high-variance: the best grouped-CV mean is 14.027 +/- 9.436 m, and fold medians range from 6.198 m to 44.500 m. Each fold contains only 3 test sessions, so variance is structural.

**Methodology checks**
No blocking inconsistencies were found in the metrics tables; prediction support aligns with the test windows used for evaluation. The no-densify ablation uses fewer windows (909 vs 1355), so its delta blends protocol and model effects.

**Most promising improvements**
- tighten temporal protocols (strict checks and label-sparsity-aware evaluation) before further architecture tuning;
- prioritize cross-device robustness via explicit domain adaptation or device-level normalization;
- expand or rebalance sessions to reduce fold variance and stabilize conclusions.
"""


SEQ_MODELS: Tuple[str, ...] = ("LSTM_FE", "GRU_FE", "LSTM_FE_KF", "GRU_FE_KF")


def clean_table(
    df: pd.DataFrame | None,
    *,
    fill_value: str = "n/a",
    drop_constant: bool = False,
) -> pd.DataFrame | None:
    if df is None:
        return None
    if len(df) == 0:
        return df.copy()
    out = df.replace([np.inf, -np.inf], np.nan).copy()
    out = out.dropna(axis=1, how="all")
    if drop_constant and len(out) > 1:
        keep_cols = []
        for c in out.columns:
            nun = out[c].nunique(dropna=True)
            if nun > 1:
                keep_cols.append(c)
        out = out[keep_cols]
    if fill_value is not None:
        out = out.fillna(fill_value)
    return out


def display_table(
    df: pd.DataFrame | None,
    *,
    fill_value: str = "n/a",
    drop_constant: bool = False,
) -> None:
    from IPython.display import display

    cleaned = clean_table(df, fill_value=fill_value, drop_constant=drop_constant)
    if cleaned is None:
        return
    display(cleaned)


def display_heading(text: str, *, level: int = 3) -> None:
    from IPython.display import display, Markdown

    level = max(1, min(int(level), 6))
    display(Markdown(f"{'#' * level} {text}"))


def config_summary(cfg: Config) -> pd.DataFrame:
    keys = [
        "run_id",
        "random_seed",
        "test_size",
        "window_size",
        "seq_min_window_size",
        "seq_min_test_windows",
        "seq_min_coverage",
        "seq_densify_step_ms",
        "seq_ignore_time_checks",
        "seq_use_topk_corr",
        "seq_topk_corr_k",
        "seq_use_pca",
        "seq_pca_n_components",
        "seq_pca_topk_corr",
        "baseline_max_speed_mps",
        "gap_thr_ms",
        "dt_max_gap_ms",
        "merge_direction",
        "merge_tolerance_ms",
        "max_nan_ratio",
        "force_recompute",
    ]
    rows = []
    for k in keys:
        if hasattr(cfg, k):
            rows.append({"parameter": k, "value": getattr(cfg, k)})
    return pd.DataFrame(rows)


def setup_plot_style(style: str = "ggplot") -> None:
    import matplotlib.pyplot as plt

    plt.style.use(style)


def load_raw_inventory(cfg: Config, raw_dir, *, force: bool = False):
    from pathlib import Path
    from IPython.display import display

    from io_data import load_or_download_dataset

    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    dfs = load_or_download_dataset(cfg.zip_url, raw_path, force=force)
    display_heading("Raw data inventory", level=3)

    rows = []
    for name, df in dfs.items():
        rows.append(
            {
                "file": name,
                "n_rows": int(len(df)),
                "n_cols": int(df.shape[1]),
                "missing_total": int(df.isna().sum().sum()),
                "has_timestamp": "Timestamp" in df.columns,
                "has_label_xy": {"label_X", "label_Y"}.issubset(df.columns),
                "has_device": "device" in df.columns,
                "has_motion": "motion" in df.columns,
            }
        )

    inventory_df = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
    display_table(inventory_df)
    summary = pd.DataFrame(
        [
            {
                "n_files": int(len(dfs)),
                "total_rows": int(inventory_df["n_rows"].sum()),
                "total_missing": int(inventory_df["missing_total"].sum()),
            }
        ]
    )
    display_table(summary)

    return dfs, inventory_df


def build_base_dataframe_report(dfs: Dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    from IPython.display import display

    from preprocessing import make_base_dataframe

    base_df = make_base_dataframe(
        dfs,
        max_nan_ratio=cfg.max_nan_ratio,
        merge_direction=cfg.merge_direction,
        merge_tolerance_ms=cfg.merge_tolerance_ms,
        dt_max_gap_ms=cfg.dt_max_gap_ms,
        gap_thr_ms=cfg.gap_thr_ms,
    )

    summary = pd.DataFrame(
        [
            {
                "rows": int(base_df.shape[0]),
                "cols": int(base_df.shape[1]),
                "sessions": int(base_df["session_id"].nunique()) if "session_id" in base_df.columns else 0,
                "segments": int(base_df[["session_id", "segment_id"]].drop_duplicates().shape[0])
                if {"session_id", "segment_id"}.issubset(base_df.columns)
                else 0,
            }
        ]
    )
    display_heading("Base dataframe summary", level=3)
    display_table(summary)
    base_cols = [
        "t_ms",
        "session_id",
        "segment_id",
        "device",
        "motion",
        "label_X",
        "label_Y",
    ]
    base_cols = [c for c in base_cols if c in base_df.columns]
    imu_cols = [c for c in cfg.imu_cols if c in base_df.columns]
    view_cols = base_cols + imu_cols
    if view_cols:
        display_heading("Base dataframe preview", level=4)
        display_table(base_df[view_cols].head(3))
    else:
        display_table(base_df.head(3))

    return base_df


def structural_integrity_report(base_df: pd.DataFrame) -> Dict[str, int]:
    checks = {
        "duplicate_columns": int(base_df.columns.duplicated().sum()),
        "duplicate_rows": int(base_df.duplicated().sum()),
        "rows_missing_t_ms": int(base_df["t_ms"].isna().sum()) if "t_ms" in base_df.columns else -1,
        "rows_missing_label": int(base_df[["label_X", "label_Y"]].isna().any(axis=1).sum()),
    }
    display_heading("Structural integrity checks", level=3)
    display_table(pd.DataFrame([checks]))

    dup_ts = base_df.groupby(["session_id", "t_ms"]).size()
    dup_summary = pd.DataFrame([{"session_t_ms_duplicates": int((dup_ts > 1).sum())}])
    display_table(dup_summary)

    return checks


def display_sanity_report(base_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    from IPython.display import display

    time_stats, global_stats = sanity_report(base_df)
    display_heading("Per-session temporal quality", level=4)
    display_table(time_stats, drop_constant=True)
    display_heading("Global plausibility stats", level=4)
    display_table(global_stats.to_frame("value"))
    return time_stats, global_stats


def missingness_and_temporal_report(base_df: pd.DataFrame, cfg: Config, *, save_path) -> Dict[str, Any]:
    from IPython.display import display
    import matplotlib.pyplot as plt

    display_heading("Missingness profile", level=3)
    missing_ratio = base_df.isna().mean().sort_values(ascending=False)
    display_table(missing_ratio.head(15).to_frame("missing_ratio"))

    seg_lens = (
        base_df.sort_values(["session_id", "segment_id", "t_ms"])
        .groupby(["session_id", "segment_id"])
        .size()
        .rename("seg_len")
    )

    display_heading("Segment length summary", level=3)
    display_table(seg_lens.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).to_frame("seg_len"))
    seg_summary = pd.DataFrame(
        [
            {
                "segments_ge_window": int((seg_lens >= cfg.window_size).sum()),
                "total_segments": int(len(seg_lens)),
                "max_seg_len": int(seg_lens.max()),
            }
        ]
    )
    display_table(seg_summary)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(seg_lens, bins=35, color="#4C72B0", alpha=0.85)
    ax[0].axvline(cfg.window_size, color="red", linestyle="--", label=f"window={cfg.window_size}")
    ax[0].set_title("Segment length distribution")
    ax[0].set_xlabel("points per segment")
    ax[0].set_ylabel("count")
    ax[0].legend()

    sess_dt = base_df.groupby("session_id")["dt_ms"].median().dropna()
    ax[1].hist(sess_dt, bins=20, color="#DD8452", alpha=0.85)
    ax[1].set_title("Median dt per session")
    ax[1].set_xlabel("dt (ms)")
    ax[1].set_ylabel("count")

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.show()

    return {"missing_ratio": missing_ratio, "seg_lens": seg_lens}


def plot_imu_corr_report(base_df: pd.DataFrame, cfg: Config, *, save_path):
    from plotting import plot_topk_corr_heatmap

    candidate_cols = [c for c in cfg.imu_cols if c in base_df.columns]
    corr_info = plot_topk_corr_heatmap(
        base_df,
        target_cols=("label_X", "label_Y"),
        candidate_cols=candidate_cols,
        topk_k=None,
        title="IMU sensor correlations vs labels",
        save_path=save_path,
    )

    if corr_info is None:
        display_heading("IMU correlation heatmap", level=3)
        display_table(pd.DataFrame([{"status": "not available"}]))
    else:
        selected = corr_info.get("selected", [])
        scores = corr_info.get("scores", {})
        rows = [{"feature": f, "score": scores.get(f, np.nan)} for f in selected]
        display_heading("IMU correlation features", level=4)
        display_table(pd.DataFrame(rows))


def split_and_report(base_df: pd.DataFrame, cfg: Config):
    from IPython.display import display

    from splitting import group_split

    df_train, df_test = group_split(
        base_df,
        group_col="session_id",
        test_size=cfg.test_size,
        seed=cfg.random_seed,
    )

    split_stats = pd.DataFrame(
        [
            {
                "subset": "train",
                "rows": len(df_train),
                "sessions": df_train.session_id.nunique(),
                "segments": df_train[["session_id", "segment_id"]].drop_duplicates().shape[0],
            },
            {
                "subset": "test",
                "rows": len(df_test),
                "sessions": df_test.session_id.nunique(),
                "segments": df_test[["session_id", "segment_id"]].drop_duplicates().shape[0],
            },
        ]
    )
    display_table(split_stats)

    display_heading("Device balance (ratio)", level=4)
    display_table(
        pd.concat(
            [
                df_train["device"].value_counts(normalize=True).rename("train_ratio"),
                df_test["device"].value_counts(normalize=True).rename("test_ratio"),
            ],
            axis=1,
        ).fillna(0)
    )

    display_heading("Motion balance (ratio, top 15)", level=4)
    motion_balance = pd.concat(
        [
            df_train["motion"].value_counts(normalize=True).rename("train_ratio"),
            df_test["motion"].value_counts(normalize=True).rename("test_ratio"),
        ],
        axis=1,
    ).fillna(0)
    display_table(motion_balance.head(15))

    return df_train, df_test


def prepare_seq_bundle_report(df_train: pd.DataFrame, df_test: pd.DataFrame, cfg: Config) -> Dict[str, Any]:
    from IPython.display import display

    seq_bundle = evaluation.prepare_seq_data(
        df_train,
        df_test,
        cfg=cfg,
        run_id=cfg.run_id,
        force_recompute=cfg.force_recompute,
    )

    seq_summary = pd.DataFrame(
        [
            {
                "n_total_test": int(seq_bundle["n_total_test"]),
                "n_test_windows": int(seq_bundle["n_test_windows"]),
                "seq_coverage": round(float(seq_bundle["seq_coverage"]), 4),
                "window_size_effective": int(seq_bundle["window_size_effective"]),
                "n_features": int(len(seq_bundle["feature_cols_seq"])),
            }
        ]
    )
    display_heading("Sequence bundle summary", level=3)
    display_table(seq_summary)
    feature_df = pd.DataFrame({"feature": seq_bundle["feature_cols_seq"][:20]})
    display_heading("First 20 sequence features", level=4)
    display_table(feature_df)

    diag = seq_bundle.get("seq_diagnostics", {})
    if diag:
        display_heading("Sequence diagnostics", level=4)
        display_table(pd.DataFrame([diag]))

    attempts = seq_bundle.get("seq_build_attempts", [])
    if attempts:
        rows = []
        for i, att in enumerate(attempts, start=1):
            tr = att.get("train_stats", {})
            te = att.get("test_stats", {})
            rows.append(
                {
                    "attempt": i,
                    "window_size": att.get("window_size"),
                    "max_dt_ms": att.get("max_dt_ms"),
                    "max_window_ms": att.get("max_window_ms"),
                    "train_windows": att.get("n_train_windows"),
                    "test_windows": att.get("n_test_windows"),
                    "coverage": att.get("coverage"),
                    "train_acceptance": tr.get("acceptance_ratio"),
                    "test_acceptance": te.get("acceptance_ratio"),
                    "test_rej_non_positive_dt": te.get("rejected_non_positive_dt"),
                    "test_rej_max_dt": te.get("rejected_max_dt"),
                    "test_rej_max_window": te.get("rejected_max_window"),
                }
            )
        display_heading("Window build attempts", level=4)
        display_table(pd.DataFrame(rows), drop_constant=True)

    return seq_bundle


def window_coverage_report(seq_bundle: Dict[str, Any], cfg: Config, *, save_path):
    from IPython.display import display
    import matplotlib.pyplot as plt

    idx_test = np.asarray(seq_bundle["idx_seq_test"], dtype=int)
    df_test_fe = seq_bundle["df_test_fe"].reset_index(drop=True)
    meta_w = df_test_fe.loc[idx_test, ["session_id", "segment_id"]]

    win_per_seg = meta_w.value_counts().rename("n_windows").reset_index()
    win_per_sess = meta_w["session_id"].value_counts().rename("n_windows").reset_index()

    display_table(win_per_seg.head(15), drop_constant=True)
    display_table(win_per_sess.head(10), drop_constant=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(win_per_seg["n_windows"], bins=30, color="#55A868", alpha=0.85)
    ax[0].set_title("Windows per segment")
    ax[0].set_xlabel("n_windows")

    ax[1].hist(win_per_sess["n_windows"], bins=20, color="#C44E52", alpha=0.85)
    ax[1].set_title("Windows per session")
    ax[1].set_xlabel("n_windows")

    for a in ax:
        a.set_ylabel("count")

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.show()


def evaluate_baselines_and_models(seq_bundle: Dict[str, Any], preds_seq_pointwise: Dict[str, np.ndarray], cfg: Config):
    from IPython.display import display

    df_eval = seq_bundle["df_test_fe"].reset_index(drop=True)
    y_true_eval = df_eval[["label_X", "label_Y"]].to_numpy(dtype=float)

    pred_last = evaluation.baseline_last_position(df_eval, time_col=cfg.time_col)
    pred_cv_oracle = evaluation.baseline_constant_velocity(
        df_eval,
        time_col=cfg.time_col,
        max_speed_mps=float(cfg.baseline_max_speed_mps),
        max_dt_ms=float(cfg.gap_thr_ms),
    )
    pred_cv_rollout = evaluation.baseline_constant_velocity_rollout(
        df_eval,
        time_col=cfg.time_col,
        max_speed_mps=float(cfg.baseline_max_speed_mps),
        max_dt_ms=float(cfg.gap_thr_ms),
    )

    pred_map = {
        "baseline_last_position": pred_last,
        "baseline_constant_velocity_oracle": pred_cv_oracle,
        "baseline_constant_velocity_rollout": pred_cv_rollout,
    }
    pred_map.update({k: v for k, v in preds_seq_pointwise.items() if isinstance(v, np.ndarray)})

    rows = []
    for name, pred in pred_map.items():
        r = evaluation.eval_fair(name, y_true_eval, pred, thresholds=cfg.thresholds)
        if r is not None:
            rows.append(r)

    metrics_all = pd.DataFrame(rows).sort_values("median_err_m").reset_index(drop=True)
    metrics_all_view = metrics_all.drop(columns=["errors_radial_m"], errors="ignore")
    cols = [
        "model",
        "model_family",
        "deployment_mode",
        "median_err_m",
        "p90_err_m",
        "rmse_2d_m",
        "mae_2d_m",
        "p_err_lt_1.0m",
        "p_err_lt_2.0m",
        "n_eval",
        "coverage_eval",
    ]
    view_cols = [c for c in cols if c in metrics_all_view.columns]
    display_table(metrics_all_view[view_cols], drop_constant=True)

    display_heading("Quick read by deployment mode", level=4)
    if len(metrics_all_view):
        agg = metrics_all_view.groupby(["model_family", "deployment_mode"], dropna=False)[
            ["median_err_m", "p90_err_m", "rmse_2d_m", "n_eval", "coverage_eval"]
        ].mean()
        display_table(agg, drop_constant=True)

    return df_eval, y_true_eval, pred_map, metrics_all


def eval_seq_models_report(seq_bundle: Dict[str, Any], cfg: Config):
    from IPython.display import display

    metrics_seq, preds_seq_pointwise, preds_seq = evaluation.eval_seq_models(seq_bundle, cfg=cfg)
    metrics_seq_view = metrics_seq.drop(columns=["errors_radial_m"], errors="ignore")
    metrics_seq_view = metrics_seq_view.sort_values("median_err_m").reset_index(drop=True)
    cols = [
        "model",
        "n",
        "median_err_m",
        "p90_err_m",
        "rmse_2d_m",
        "mae_2d_m",
        "p_err_lt_1.0m",
        "p_err_lt_2.0m",
    ]
    view_cols = [c for c in cols if c in metrics_seq_view.columns]
    display_table(metrics_seq_view[view_cols], drop_constant=True)
    return metrics_seq, preds_seq_pointwise, preds_seq


def rollout_comparison_report(metrics_all: pd.DataFrame, *, ref_name: str = "baseline_constant_velocity_rollout"):
    from IPython.display import display

    if ref_name in metrics_all["model"].values:
        ref_med = float(metrics_all.loc[metrics_all["model"] == ref_name, "median_err_m"].iloc[0])
        comp = metrics_all[
            [
                "model",
                "model_family",
                "deployment_mode",
                "median_err_m",
                "p90_err_m",
                "rmse_2d_m",
                "n_eval",
                "coverage_eval",
            ]
        ].copy()
        comp["delta_median_vs_rollout_m"] = comp["median_err_m"] - ref_med
        display_table(comp.sort_values("delta_median_vs_rollout_m"), drop_constant=True)
        return comp

    display_heading("Rollout baseline not found.", level=4)
    return pd.DataFrame()


def protocol_metrics_report(df_eval: pd.DataFrame, pred_map: Dict[str, np.ndarray], cfg: Config) -> pd.DataFrame:
    from IPython.display import display

    metrics_protocols = evaluation.evaluate_predictions_protocols(
        df_eval,
        pred_map,
        thresholds=cfg.thresholds,
    )

    if len(metrics_protocols) == 0:
        display_heading("No protocol metrics are available.", level=4)
        return metrics_protocols

    view = metrics_protocols.drop(columns=["errors_radial_m"], errors="ignore")
    cols = [
        "protocol",
        "model",
        "model_family",
        "deployment_mode",
        "median_err_m",
        "p90_err_m",
        "rmse_2d_m",
        "n_eval",
        "coverage_eval",
    ]
    view_cols = [c for c in cols if c in view.columns]
    display_table(view[view_cols].sort_values(["median_err_m", "model"]), drop_constant=True)

    seq_view = view[view["model_family"] == "sequential_dl"].copy()
    if len(seq_view):
        display_heading("Sequential models under the dense protocol", level=4)
        display_table(
            seq_view[["model", "median_err_m", "p90_err_m", "rmse_2d_m", "n_eval", "coverage_eval"]].sort_values(
                "median_err_m"
            ),
            drop_constant=True,
        )

    return metrics_protocols


def failure_uncertainty_report(
    df_eval: pd.DataFrame,
    pred_map: Dict[str, np.ndarray],
    metrics_protocols: pd.DataFrame,
    *,
    cfg: Config,
):
    from IPython.display import display

    report_pack = run_failure_uncertainty_bootstrap(
        df_eval,
        pred_map,
        metrics_protocols,
        cfg=cfg,
    )

    failure_pack = report_pack["failure_pack"]
    uncertainty_df = report_pack["uncertainty_df"]
    uncertainty_summary = report_pack["uncertainty_summary"]
    calibration_df = report_pack["calibration_df"]
    comparison_df = report_pack["comparison_df"]
    comparison_result = report_pack["comparison_result"]

    if len(failure_pack.get("by_device_motion", pd.DataFrame())):
        display_heading("Top device/motion contexts (lowest median error)", level=4)
        display_table(
            failure_pack["by_device_motion"].sort_values(["model", "median_err_m"]).groupby("model").head(6),
            drop_constant=True,
        )

    if len(failure_pack.get("by_dt_bin", pd.DataFrame())):
        display_heading("Sensitivity to local temporal interval (dt bins)", level=4)
        display_table(
            failure_pack["by_dt_bin"].sort_values(["model", "median_err_m"]).groupby("model").head(8),
            drop_constant=True,
        )

    if len(uncertainty_df):
        display_heading("Uncertainty summary (inter-model disagreement)", level=4)
        display_table(pd.DataFrame([uncertainty_summary]))
        if len(calibration_df):
            display_heading("Empirical calibration: error vs uncertainty quantile", level=4)
            display_table(calibration_df, drop_constant=True)

    if len(comparison_df):
        display_table(comparison_df, drop_constant=True)
    else:
        display_heading("Bootstrap comparison between sequential model and rollout baseline is not available.", level=4)

    return failure_pack, uncertainty_df, uncertainty_summary, calibration_df, comparison_df, comparison_result


def display_summary_lines(
    *,
    cfg: Config,
    seq_bundle: Dict[str, Any],
    metrics_protocols: pd.DataFrame,
    df_eval: pd.DataFrame,
    comparison_result: Dict[str, Any] | None,
    ablation_df: pd.DataFrame | None,
    uncertainty_summary: Dict[str, Any] | None,
    cross_df: pd.DataFrame | None,
    cv_summary_df: pd.DataFrame | None,
    cv_meta_df: pd.DataFrame | None,
    cv_folds_df: pd.DataFrame | None,
    metrics_seq: pd.DataFrame | None,
    preds_seq_pointwise: Dict[str, np.ndarray] | None,
):
    from IPython.display import display, Markdown

    summary_lines = build_key_summary_lines(
        cfg=cfg,
        seq_bundle=seq_bundle,
        metrics_protocols=metrics_protocols,
        df_eval=df_eval,
        comparison_result=comparison_result,
        ablation_df=ablation_df,
        uncertainty_summary=uncertainty_summary,
        cross_df=cross_df,
        cv_summary_df=cv_summary_df,
        cv_meta_df=cv_meta_df,
        cv_folds_df=cv_folds_df,
        metrics_seq=metrics_seq,
        preds_seq_pointwise=preds_seq_pointwise,
    )

    summary_text = "\n".join([f"- {s}" for s in summary_lines])
    display(Markdown(summary_text))
    return summary_lines


def ablation_cv_report(
    base_df: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    cfg: Config,
    run_ablation: bool = True,
    run_topk_toggle_ablation: bool = True,
    run_pca_toggle_ablation: bool = False,
    run_cv: bool = True,
    cv_splits: int = 5,
):
    from IPython.display import display

    ablation_pack = run_ablation_and_cv(
        base_df,
        df_train,
        df_test,
        cfg=cfg,
        run_ablation=run_ablation,
        run_topk_toggle_ablation=run_topk_toggle_ablation,
        run_pca_toggle_ablation=run_pca_toggle_ablation,
        run_cv=run_cv,
        cv_splits=cv_splits,
    )

    ablation_df = ablation_pack["ablation_df"]
    cv_folds_df = ablation_pack["cv_folds_df"]
    cv_summary_df = ablation_pack["cv_summary_df"]
    cv_meta_df = ablation_pack["cv_meta_df"]

    if run_ablation:
        if len(ablation_df):
            view_cols = [
                "tag",
                "best_model",
                "best_median_err_m",
                "best_p90_err_m",
                "n_features",
                "n_test_windows",
                "coverage",
                "window_size",
                "densify_step_ms",
                "seq_use_topk_corr",
                "seq_topk_corr_k",
                "seq_use_pca",
                "seq_pca_n_components",
                "seq_pca_topk_corr",
                "status",
            ]
            view_cols = [c for c in view_cols if c in ablation_df.columns]
            view_df = ablation_df[view_cols].sort_values(["status", "best_median_err_m"], na_position="last")
            display_table(view_df, drop_constant=True)
        if ablation_pack.get("ablation_notes"):
            notes_df = pd.DataFrame({"note": ablation_pack["ablation_notes"]})
            display_heading("Ablation notes", level=4)
            display_table(notes_df)
    else:
        display_heading("Ablation skipped (run_ablation=False).", level=4)

    if run_cv:
        display_heading("CV meta", level=4)
        meta_cols = ["fold", "seed", "test_sessions", "test_rows", "test_windows", "test_coverage", "window_size_effective"]
        meta_cols = [c for c in meta_cols if c in cv_meta_df.columns]
        display_table(cv_meta_df[meta_cols], drop_constant=True)
        if len(cv_summary_df):
            display_heading("Cross-split CV summary (sequential models)", level=4)
            cv_cols = [
                "model",
                "n_folds",
                "median_err_mean",
                "median_err_std",
                "p90_err_mean",
                "p90_err_std",
                "rmse_mean",
                "rmse_std",
            ]
            cv_cols = [c for c in cv_cols if c in cv_summary_df.columns]
            display_table(cv_summary_df[cv_cols], drop_constant=True)
    else:
        display_heading("Grouped cross-split CV is disabled (run_cv=False).", level=4)

    return ablation_df, cv_folds_df, cv_summary_df, cv_meta_df


def cross_device_report(base_df: pd.DataFrame, *, cfg: Config, run_cross_device: bool = True):
    from IPython.display import display

    cross_pack = run_cross_device_transfer(
        base_df,
        cfg=cfg,
        run_cross_device=run_cross_device,
    )

    if cross_pack.get("messages"):
        display_heading("Cross-device run log", level=4)
        display_table(pd.DataFrame({"message": cross_pack["messages"]}))

    cross_df = cross_pack["cross_df"]
    cross_results = cross_pack["cross_results"]

    if len(cross_df):
        cols = [
            "pair",
            "model",
            "median_err_m",
            "p90_err_m",
            "rmse_2d_m",
            "n",
            "test_windows",
            "test_coverage",
            "window_size_eff",
        ]
        display_heading("Cross-device summary", level=4)
        display_table(cross_df[cols], drop_constant=True)
    else:
        display_heading("No cross-device results were generated.", level=4)

    return cross_df, cross_results


def export_reports(
    *,
    report_dir,
    run_id: str,
    metrics_all: pd.DataFrame | None = None,
    metrics_protocols: pd.DataFrame | None = None,
    ablation_df: pd.DataFrame | None = None,
    cross_df: pd.DataFrame | None = None,
    cv_summary_df: pd.DataFrame | None = None,
    cv_folds_df: pd.DataFrame | None = None,
    cv_meta_df: pd.DataFrame | None = None,
    hypothesis_df: pd.DataFrame | None = None,
    summary_lines: Sequence[str] | None = None,
    conclusion_text: str | None = None,
) -> None:
    from pathlib import Path

    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write_df(df: pd.DataFrame | None, name: str) -> None:
        if df is None or len(df) == 0:
            return
        out = out_dir / f"{name}__{run_id}.csv"
        cleaned = clean_table(df.drop(columns=["errors_radial_m"], errors="ignore"))
        if cleaned is None or len(cleaned) == 0:
            return
        cleaned.to_csv(out, index=False)

    _write_df(metrics_all, "metrics_all")
    _write_df(metrics_protocols, "metrics_protocols")
    _write_df(ablation_df, "ablation")
    _write_df(cross_df, "cross_device")
    _write_df(cv_summary_df, "cv_summary")
    _write_df(cv_folds_df, "cv_folds")
    _write_df(cv_meta_df, "cv_meta")
    _write_df(hypothesis_df, "hypothesis")

    if summary_lines:
        (out_dir / f"summary_lines__{run_id}.md").write_text("\n".join([f"- {s}" for s in summary_lines]))

    if conclusion_text:
        (out_dir / f"conclusion__{run_id}.md").write_text(conclusion_text)

    print(f"Reports exported to {out_dir}")

def sanity_report(df, session_col="session_id", t_col="t_ms", x_col="label_X", y_col="label_Y"):
    d = df.sort_values([session_col, t_col]).copy()

    def per_session(g):
        t = g[t_col].to_numpy(dtype=float)
        dt = np.diff(t)
        return pd.Series({
            "n": len(g),
            "bad_dt_le0": int(np.sum(dt <= 0)),
            "bad_dt_le0_ratio": float(np.mean(dt <= 0)) if len(dt) else 0.0,
            "dt_median_ms": float(np.median(dt)) if len(dt) else np.nan,
            "dt_p95_ms": float(np.quantile(dt, 0.95)) if len(dt) else np.nan,
            "dt_max_ms": float(np.max(dt)) if len(dt) else np.nan,
            "gaps_gt_500ms": int(np.sum(dt > 500)),
            "gaps_gt_1000ms": int(np.sum(dt > 1000)),
        })

    time_stats = d.groupby(session_col).apply(per_session).sort_values("dt_p95_ms", ascending=False)

    dx = d.groupby(session_col)[x_col].diff()
    dy = d.groupby(session_col)[y_col].diff()
    step = np.sqrt(dx**2 + dy**2)
    dt_s = d.groupby(session_col)[t_col].diff() / 1000.0
    vel = (step / dt_s).replace([np.inf, -np.inf], np.nan)

    imu_cols = [c for c in ["AccelX", "AccelY", "AccelZ", "GyroX", "GyroY", "GyroZ"] if c in d.columns]
    if imu_cols:
        imu_norm = np.sqrt(np.square(d[imu_cols].astype(float)).sum(axis=1))
        corr_imu_step = pd.Series(imu_norm).corr(step)
        corr_imu_vel = pd.Series(imu_norm).corr(vel)
    else:
        corr_imu_step = np.nan
        corr_imu_vel = np.nan

    global_stats = pd.Series({
        "x_min": float(d[x_col].min()),
        "x_max": float(d[x_col].max()),
        "x_std": float(d[x_col].std()),
        "y_min": float(d[y_col].min()),
        "y_max": float(d[y_col].max()),
        "y_std": float(d[y_col].std()),
        "step_median_m": float(step.median()),
        "step_p95_m": float(step.quantile(0.95)),
        "speed_median_mps": float(vel.median()),
        "speed_p95_mps": float(vel.quantile(0.95)),
        "speed_max_mps": float(vel.max()),
        "corr_imu_step": float(corr_imu_step) if pd.notna(corr_imu_step) else np.nan,
        "corr_imu_vel": float(corr_imu_vel) if pd.notna(corr_imu_vel) else np.nan,
    })

    return time_stats, global_stats


def _bootstrap_delta_median(
    y_true: np.ndarray,
    y_seq: np.ndarray,
    y_ref: np.ndarray,
    *,
    n_boot: int = 400,
    seed: int = 42,
) -> Dict[str, float] | None:
    y_true = np.asarray(y_true, dtype=float)
    y_seq = np.asarray(y_seq, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    n = min(len(y_true), len(y_seq), len(y_ref))
    if n == 0:
        return None

    yt = y_true[:n]
    ys = y_seq[:n]
    yr = y_ref[:n]
    mask = np.isfinite(yt).all(axis=1) & np.isfinite(ys).all(axis=1) & np.isfinite(yr).all(axis=1)
    if int(mask.sum()) < 20:
        return None

    err_seq = np.linalg.norm(ys[mask] - yt[mask], axis=1)
    err_ref = np.linalg.norm(yr[mask] - yt[mask], axis=1)
    rng = np.random.default_rng(int(seed))
    diffs = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(err_seq), len(err_seq))
        diffs.append(float(np.median(err_seq[idx]) - np.median(err_ref[idx])))
    diffs = np.asarray(diffs, dtype=float)

    return {
        "delta_median_seq_minus_rollout": float(np.median(err_seq) - np.median(err_ref)),
        "ci95_low": float(np.quantile(diffs, 0.025)),
        "ci95_high": float(np.quantile(diffs, 0.975)),
        "n_common": int(mask.sum()),
    }


def run_failure_uncertainty_bootstrap(
    df_eval: pd.DataFrame,
    pred_map: Dict[str, np.ndarray],
    metrics_protocols: pd.DataFrame,
    *,
    cfg: Config,
    seq_models: Sequence[str] = SEQ_MODELS,
    n_boot: int = 400,
) -> Dict[str, Any]:
    """Run context-failure analysis, uncertainty diagnostics, and bootstrap comparison."""

    failure_pack = evaluation.failure_analysis_by_context(
        df_eval,
        pred_map,
        time_col=str(cfg.time_col),
    )

    y_true_eval = df_eval[["label_X", "label_Y"]].to_numpy(dtype=float)
    seq_for_uncertainty = {k: v for k, v in pred_map.items() if k in set(seq_models)}
    uncertainty_df, uncertainty_summary = evaluation.estimate_ensemble_uncertainty(
        seq_for_uncertainty,
        y_true=y_true_eval,
    )

    calibration_df = pd.DataFrame()
    if len(uncertainty_df) and "error_mean_pred_m" in uncertainty_df.columns:
        q = uncertainty_df.copy()
        q = q[np.isfinite(q["uncertainty_m"]) & np.isfinite(q["error_mean_pred_m"])].copy()
        if len(q) >= 20:
            q["u_bin"] = pd.qcut(q["uncertainty_m"], q=5, labels=False, duplicates="drop")
            calibration_df = q.groupby("u_bin", as_index=False).agg(
                n=("idx", "size"),
                uncertainty_mean=("uncertainty_m", "mean"),
                error_mean=("error_mean_pred_m", "mean"),
                error_median=("error_mean_pred_m", "median"),
            )

    comparison_rows = []
    comparison_result = None
    if len(metrics_protocols):
        m_proto = metrics_protocols[
            (metrics_protocols["protocol"] == "dense_aligned")
            & (metrics_protocols["model"].isin(seq_models))
        ].sort_values("median_err_m")
        if len(m_proto) and ("baseline_constant_velocity_rollout" in pred_map):
            best_seq = str(m_proto.iloc[0]["model"])
            res = _bootstrap_delta_median(
                y_true_eval,
                np.asarray(pred_map[best_seq], dtype=float),
                np.asarray(pred_map["baseline_constant_velocity_rollout"], dtype=float),
                n_boot=int(n_boot),
                seed=int(cfg.random_seed),
            )
            if res is not None:
                comparison_rows.append({"protocol": "dense_aligned", "best_seq": best_seq, **res})

    comparison_df = pd.DataFrame(comparison_rows)
    if len(comparison_df):
        comparison_result = comparison_df.iloc[0].to_dict()

    return {
        "failure_pack": failure_pack,
        "uncertainty_df": uncertainty_df,
        "uncertainty_summary": uncertainty_summary,
        "calibration_df": calibration_df,
        "comparison_df": comparison_df,
        "comparison_result": comparison_result,
    }


def build_ablation_grid(
    cfg: Config,
    *,
    run_topk_toggle_ablation: bool = True,
    run_pca_toggle_ablation: bool = False,
) -> List[Dict[str, Any]]:
    """Build the standard ablation grid from notebook toggles."""

    grid: List[Dict[str, Any]] = [
        {
            "tag": "base",
            "window_size": cfg.window_size,
            "seq_densify_step_ms": cfg.seq_densify_step_ms,
            "seq_ignore_time_checks": cfg.seq_ignore_time_checks,
            "seq_use_topk_corr": cfg.seq_use_topk_corr,
            "seq_topk_corr_k": cfg.seq_topk_corr_k,
            "seq_use_pca": cfg.seq_use_pca,
            "seq_pca_n_components": cfg.seq_pca_n_components,
            "seq_pca_topk_corr": cfg.seq_pca_topk_corr,
        },
        {
            "tag": "no_densify",
            "window_size": cfg.window_size,
            "seq_densify_step_ms": None,
            "seq_ignore_time_checks": cfg.seq_ignore_time_checks,
            "seq_use_topk_corr": cfg.seq_use_topk_corr,
            "seq_topk_corr_k": cfg.seq_topk_corr_k,
            "seq_use_pca": cfg.seq_use_pca,
            "seq_pca_n_components": cfg.seq_pca_n_components,
            "seq_pca_topk_corr": cfg.seq_pca_topk_corr,
        },
        {
            "tag": "strict_time",
            "window_size": cfg.window_size,
            "seq_densify_step_ms": cfg.seq_densify_step_ms,
            "seq_ignore_time_checks": False,
            "seq_use_topk_corr": cfg.seq_use_topk_corr,
            "seq_topk_corr_k": cfg.seq_topk_corr_k,
            "seq_use_pca": cfg.seq_use_pca,
            "seq_pca_n_components": cfg.seq_pca_n_components,
            "seq_pca_topk_corr": cfg.seq_pca_topk_corr,
        },
        {
            "tag": "short_window",
            "window_size": max(cfg.seq_min_window_size, 8),
            "seq_densify_step_ms": cfg.seq_densify_step_ms,
            "seq_ignore_time_checks": cfg.seq_ignore_time_checks,
            "seq_use_topk_corr": cfg.seq_use_topk_corr,
            "seq_topk_corr_k": cfg.seq_topk_corr_k,
            "seq_use_pca": cfg.seq_use_pca,
            "seq_pca_n_components": cfg.seq_pca_n_components,
            "seq_pca_topk_corr": cfg.seq_pca_topk_corr,
        },
        {
            "tag": "long_window",
            "window_size": max(cfg.window_size, 16),
            "seq_densify_step_ms": cfg.seq_densify_step_ms,
            "seq_ignore_time_checks": cfg.seq_ignore_time_checks,
            "seq_use_topk_corr": cfg.seq_use_topk_corr,
            "seq_topk_corr_k": cfg.seq_topk_corr_k,
            "seq_use_pca": cfg.seq_use_pca,
            "seq_pca_n_components": cfg.seq_pca_n_components,
            "seq_pca_topk_corr": cfg.seq_pca_topk_corr,
        },
    ]

    if run_topk_toggle_ablation:
        grid.extend(
            [
                {
                    "tag": "topk_off",
                    "window_size": cfg.window_size,
                    "seq_densify_step_ms": cfg.seq_densify_step_ms,
                    "seq_ignore_time_checks": cfg.seq_ignore_time_checks,
                    "seq_use_topk_corr": False,
                    "seq_topk_corr_k": cfg.seq_topk_corr_k,
                    "seq_use_pca": cfg.seq_use_pca,
                    "seq_pca_n_components": cfg.seq_pca_n_components,
                    "seq_pca_topk_corr": cfg.seq_pca_topk_corr,
                },
                {
                    "tag": "topk_k10",
                    "window_size": cfg.window_size,
                    "seq_densify_step_ms": cfg.seq_densify_step_ms,
                    "seq_ignore_time_checks": cfg.seq_ignore_time_checks,
                    "seq_use_topk_corr": True,
                    "seq_topk_corr_k": 10,
                    "seq_use_pca": cfg.seq_use_pca,
                    "seq_pca_n_components": cfg.seq_pca_n_components,
                    "seq_pca_topk_corr": cfg.seq_pca_topk_corr,
                },
            ]
        )

    if run_pca_toggle_ablation:
        grid.extend(
            [
                {
                    "tag": "pca_off",
                    "window_size": cfg.window_size,
                    "seq_densify_step_ms": cfg.seq_densify_step_ms,
                    "seq_ignore_time_checks": cfg.seq_ignore_time_checks,
                    "seq_use_topk_corr": False,
                    "seq_topk_corr_k": cfg.seq_topk_corr_k,
                    "seq_use_pca": False,
                    "seq_pca_n_components": cfg.seq_pca_n_components,
                    "seq_pca_topk_corr": cfg.seq_pca_topk_corr,
                },
                {
                    "tag": "pca_on",
                    "window_size": cfg.window_size,
                    "seq_densify_step_ms": cfg.seq_densify_step_ms,
                    "seq_ignore_time_checks": cfg.seq_ignore_time_checks,
                    "seq_use_topk_corr": False,
                    "seq_topk_corr_k": cfg.seq_topk_corr_k,
                    "seq_use_pca": True,
                    "seq_pca_n_components": cfg.seq_pca_n_components,
                    "seq_pca_topk_corr": cfg.seq_pca_topk_corr,
                },
                {
                    "tag": "pca_n10",
                    "window_size": cfg.window_size,
                    "seq_densify_step_ms": cfg.seq_densify_step_ms,
                    "seq_ignore_time_checks": cfg.seq_ignore_time_checks,
                    "seq_use_topk_corr": False,
                    "seq_topk_corr_k": cfg.seq_topk_corr_k,
                    "seq_use_pca": True,
                    "seq_pca_n_components": 10,
                    "seq_pca_topk_corr": max(10, cfg.seq_pca_topk_corr),
                },
            ]
        )

    return grid


def run_ablation_and_cv(
    base_df: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    cfg: Config,
    run_ablation: bool = True,
    run_topk_toggle_ablation: bool = True,
    run_pca_toggle_ablation: bool = False,
    run_cv: bool = True,
    cv_splits: int = 5,
) -> Dict[str, Any]:
    """Run sequential ablation and grouped CV with notebook-friendly outputs."""

    ablation_df = pd.DataFrame()
    notes: List[str] = []

    if run_ablation:
        rows = []
        grid = build_ablation_grid(
            cfg,
            run_topk_toggle_ablation=run_topk_toggle_ablation,
            run_pca_toggle_ablation=run_pca_toggle_ablation,
        )
        for spec in grid:
            cfg_i = deepcopy(cfg)
            cfg_i.window_size = int(spec["window_size"])
            cfg_i.seq_densify_step_ms = spec["seq_densify_step_ms"]
            cfg_i.seq_ignore_time_checks = bool(spec["seq_ignore_time_checks"])
            cfg_i.seq_use_topk_corr = bool(spec.get("seq_use_topk_corr", cfg.seq_use_topk_corr))
            cfg_i.seq_topk_corr_k = int(spec.get("seq_topk_corr_k", cfg.seq_topk_corr_k))
            cfg_i.seq_use_pca = bool(spec.get("seq_use_pca", cfg.seq_use_pca))
            cfg_i.seq_pca_n_components = int(spec.get("seq_pca_n_components", cfg.seq_pca_n_components))
            cfg_i.seq_pca_topk_corr = int(spec.get("seq_pca_topk_corr", cfg.seq_pca_topk_corr))
            cfg_i.run_id = f"{cfg.run_id}__abl__{spec['tag']}"

            set_global_seed(cfg_i.random_seed)
            try:
                bundle_i = evaluation.prepare_seq_data(
                    df_train,
                    df_test,
                    cfg=cfg_i,
                    run_id=cfg_i.run_id,
                    force_recompute=cfg.force_recompute,
                )
                metrics_i, _, _ = evaluation.eval_seq_models(bundle_i, cfg=cfg_i)
                cand = metrics_i[metrics_i["model"].isin(SEQ_MODELS)].sort_values("median_err_m")
                best = cand.iloc[0]
                rows.append(
                    {
                        "tag": spec["tag"],
                        "window_size": cfg_i.window_size,
                        "densify_step_ms": cfg_i.seq_densify_step_ms,
                        "ignore_time_checks": cfg_i.seq_ignore_time_checks,
                        "seq_use_topk_corr": cfg_i.seq_use_topk_corr,
                        "seq_topk_corr_k": int(cfg_i.seq_topk_corr_k) if cfg_i.seq_use_topk_corr else 0,
                        "seq_use_pca": cfg_i.seq_use_pca,
                        "seq_pca_n_components": int(cfg_i.seq_pca_n_components) if cfg_i.seq_use_pca else 0,
                        "seq_pca_topk_corr": int(cfg_i.seq_pca_topk_corr) if cfg_i.seq_use_pca else 0,
                        "n_features": len(bundle_i["feature_cols_seq"]),
                        "n_test_windows": bundle_i["n_test_windows"],
                        "coverage": bundle_i["seq_coverage"],
                        "best_model": best["model"],
                        "best_median_err_m": best["median_err_m"],
                        "best_p90_err_m": best["p90_err_m"],
                        "status": "ok",
                        "error": "",
                    }
                )
            except Exception as e:
                rows.append(
                    {
                        "tag": spec["tag"],
                        "window_size": cfg_i.window_size,
                        "densify_step_ms": cfg_i.seq_densify_step_ms,
                        "ignore_time_checks": cfg_i.seq_ignore_time_checks,
                        "seq_use_topk_corr": cfg_i.seq_use_topk_corr,
                        "seq_topk_corr_k": int(cfg_i.seq_topk_corr_k) if cfg_i.seq_use_topk_corr else 0,
                        "seq_use_pca": cfg_i.seq_use_pca,
                        "seq_pca_n_components": int(cfg_i.seq_pca_n_components) if cfg_i.seq_use_pca else 0,
                        "seq_pca_topk_corr": int(cfg_i.seq_pca_topk_corr) if cfg_i.seq_use_pca else 0,
                        "n_features": np.nan,
                        "n_test_windows": np.nan,
                        "coverage": np.nan,
                        "best_model": "",
                        "best_median_err_m": np.nan,
                        "best_p90_err_m": np.nan,
                        "status": "fail",
                        "error": str(e),
                    }
                )
        ablation_df = pd.DataFrame(rows)

        if {"base", "topk_off"}.issubset(set(ablation_df.get("tag", pd.Series(dtype=str)))):
            ok = ablation_df[ablation_df["status"] == "ok"].set_index("tag")
            if {"base", "topk_off"}.issubset(set(ok.index)):
                delta = float(ok.loc["topk_off", "best_median_err_m"] - ok.loc["base", "best_median_err_m"])
                dfeat = int(ok.loc["topk_off", "n_features"] - ok.loc["base", "n_features"])
                notes.append(f"Top-k on/off delta (topk_off - base): {delta:.3f} m; feature count delta: {dfeat:+d}.")

        if {"pca_off", "pca_on"}.issubset(set(ablation_df.get("tag", pd.Series(dtype=str)))):
            ok = ablation_df[ablation_df["status"] == "ok"].set_index("tag")
            if {"pca_off", "pca_on"}.issubset(set(ok.index)):
                delta = float(ok.loc["pca_on", "best_median_err_m"] - ok.loc["pca_off", "best_median_err_m"])
                dfeat = int(ok.loc["pca_on", "n_features"] - ok.loc["pca_off", "n_features"])
                notes.append(f"PCA on/off delta (pca_on - pca_off): {delta:.3f} m; feature count delta: {dfeat:+d}.")

    cv_folds_df = pd.DataFrame()
    cv_summary_df = pd.DataFrame()
    cv_meta_df = pd.DataFrame()
    if run_cv:
        cv_folds_df, cv_summary_df, cv_meta_df = evaluation.run_seq_group_split_cv(
            base_df,
            cfg=cfg,
            n_splits=int(cv_splits),
            run_id_prefix=f"{cfg.run_id}",
            test_size=cfg.test_size,
            force_recompute=cfg.force_recompute,
        )

    return {
        "ablation_df": ablation_df,
        "ablation_notes": notes,
        "cv_folds_df": cv_folds_df,
        "cv_summary_df": cv_summary_df,
        "cv_meta_df": cv_meta_df,
    }


def run_cross_device_transfer(
    base_df: pd.DataFrame,
    *,
    cfg: Config,
    run_cross_device: bool = True,
) -> Dict[str, Any]:
    """Train on source device and evaluate on target device for every direction."""

    if not run_cross_device:
        return {"cross_df": pd.DataFrame(), "cross_results": [], "messages": ["Cross-device is disabled."]}

    messages: List[str] = []
    cross_rows: List[pd.DataFrame] = []
    cross_results: List[Dict[str, Any]] = []

    devices = sorted(base_df["device"].dropna().unique().tolist()) if "device" in base_df.columns else []
    if len(devices) < 2:
        return {
            "cross_df": pd.DataFrame(),
            "cross_results": [],
            "messages": ["Cross-device evaluation cannot run: fewer than 2 devices were detected."],
        }

    for src in devices:
        for tgt in devices:
            if src == tgt:
                continue
            pair_id = f"{src}->{tgt}"
            messages.append(f"[cross-device] {pair_id}")
            set_global_seed(cfg.random_seed)

            df_src = base_df[base_df["device"] == src].reset_index(drop=True)
            df_tgt = base_df[base_df["device"] == tgt].reset_index(drop=True)
            run_cd = f"{cfg.run_id}__cd__{src}_to_{tgt}"
            bundle_cd = evaluation.prepare_seq_data(
                df_src,
                df_tgt,
                cfg=cfg,
                run_id=run_cd,
                force_recompute=cfg.force_recompute,
            )
            metrics_cd, _, preds_cd = evaluation.eval_seq_models(bundle_cd, cfg=cfg)
            tmp = metrics_cd.drop(columns=["errors_radial_m"], errors="ignore").copy()
            tmp["source"] = src
            tmp["target"] = tgt
            tmp["pair"] = pair_id
            tmp["test_windows"] = bundle_cd["n_test_windows"]
            tmp["test_coverage"] = bundle_cd["seq_coverage"]
            tmp["window_size_eff"] = bundle_cd["window_size_effective"]
            cross_rows.append(tmp)

            for model_name, pack in preds_cd.items():
                cross_results.append(
                    {
                        "source": src,
                        "target": tgt,
                        "pair": pair_id,
                        "model": model_name,
                        "y_true": pack["y_true"],
                        "y_pred": pack["y_pred"],
                        "t_ms": pack.get("t_ms"),
                        "session_id": pack.get("session_id"),
                        "segment_id": pack.get("segment_id"),
                    }
                )

    if not cross_rows:
        messages.append("No cross-device results were generated.")
        return {"cross_df": pd.DataFrame(), "cross_results": cross_results, "messages": messages}

    cross_df = pd.concat(cross_rows, ignore_index=True)
    cross_df = cross_df.sort_values(["pair", "median_err_m"]).reset_index(drop=True)
    return {"cross_df": cross_df, "cross_results": cross_results, "messages": messages}


def build_hypothesis_table(
    *,
    metrics_protocols: pd.DataFrame | None = None,
    ablation_df: pd.DataFrame | None = None,
    seq_bundle: Dict[str, Any] | None = None,
    cross_df: pd.DataFrame | None = None,
    uncertainty_summary: Dict[str, Any] | None = None,
    cv_summary_df: pd.DataFrame | None = None,
    cfg: Config | None = None,
) -> pd.DataFrame:
    """Build the hypothesis/evidence/conclusion table used in the final report."""

    rows: List[Dict[str, str]] = []

    h1_evidence = "not enough info"
    h1_conclusion = "undetermined"
    if metrics_protocols is not None and len(metrics_protocols):
        dense = metrics_protocols[metrics_protocols["protocol"] == "dense_aligned"]
        seq_dense = dense[dense["model_family"] == "sequential_dl"].sort_values("median_err_m")
        rollout_dense = dense[dense["model"] == "baseline_constant_velocity_rollout"]
        if len(seq_dense) and len(rollout_dense):
            best = seq_dense.iloc[0]
            ref = rollout_dense.iloc[0]
            h1_evidence = (
                f"best seq dense={best['model']} ({best['median_err_m']:.3f} m) vs "
                f"rollout ({ref['median_err_m']:.3f} m)"
            )
            h1_conclusion = "supported" if float(best["median_err_m"]) < float(ref["median_err_m"]) else "not supported"
    rows.append(
        {
            "hypothesis": "H1 sequential beats deployable rollout (dense)",
            "evidence": h1_evidence,
            "conclusion": h1_conclusion,
        }
    )

    h2_evidence = "top-k toggle not run"
    h2_conclusion = "undetermined"
    if ablation_df is not None and len(ablation_df):
        ok = ablation_df[ablation_df["status"] == "ok"].copy()
        if len(ok) and {"base", "topk_off"}.issubset(set(ok["tag"])):
            t = ok.set_index("tag")
            delta = float(t.loc["topk_off", "best_median_err_m"] - t.loc["base", "best_median_err_m"])
            dfeat = float(t.loc["topk_off", "n_features"] - t.loc["base", "n_features"])
            h2_evidence = f"delta(topk_off-base)={delta:.3f} m, feature_delta={dfeat:+.0f}"
            h2_conclusion = "supported" if abs(delta) > 0.05 else "weak support"
    rows.append(
        {
            "hypothesis": "H2 top-k feature selection changes performance",
            "evidence": h2_evidence,
            "conclusion": h2_conclusion,
        }
    )

    h3_evidence = "sequence diagnostics unavailable"
    h3_conclusion = "undetermined"
    diag = seq_bundle.get("seq_diagnostics", {}) if isinstance(seq_bundle, dict) else {}
    attempts = seq_bundle.get("seq_build_attempts", []) if isinstance(seq_bundle, dict) else []
    if diag and isinstance(seq_bundle, dict):
        cov = seq_bundle.get("seq_coverage", np.nan)
        densify_factor = diag.get("test_densify_factor", np.nan)
        acc = np.nan
        if attempts:
            acc = attempts[-1].get("test_stats", {}).get("acceptance_ratio", np.nan)
        h3_evidence = f"coverage={cov:.3f}, test_densify_factor={densify_factor:.2f}, acceptance={acc:.3f}"
        min_cov = float(getattr(cfg, "seq_min_coverage", 0.0)) if cfg is not None else 0.0
        h3_conclusion = "supported" if cov >= min_cov else "partially supported"
    rows.append(
        {
            "hypothesis": "H3 temporal preparation quality is measurable",
            "evidence": h3_evidence,
            "conclusion": h3_conclusion,
        }
    )

    h4_evidence = "cross-device section not run"
    h4_conclusion = "undetermined"
    if cross_df is not None and len(cross_df) and metrics_protocols is not None and len(metrics_protocols):
        dense_seq = metrics_protocols[
            (metrics_protocols["protocol"] == "dense_aligned")
            & (metrics_protocols["model_family"] == "sequential_dl")
        ]
        if len(dense_seq):
            intra_med = float(dense_seq["median_err_m"].median())
            cross_seq = cross_df[cross_df["model"].isin(SEQ_MODELS)]
            if len(cross_seq):
                cross_med = float(cross_seq["median_err_m"].median())
                h4_evidence = f"intra seq median={intra_med:.3f} m, cross seq median={cross_med:.3f} m"
                h4_conclusion = "supported" if cross_med > intra_med else "not supported"
    rows.append({"hypothesis": "H4 cross-device penalty exists", "evidence": h4_evidence, "conclusion": h4_conclusion})

    h5_evidence = "uncertainty estimate unavailable"
    h5_conclusion = "undetermined"
    if uncertainty_summary is not None and len(uncertainty_summary):
        corr_ue = uncertainty_summary.get("corr_uncertainty_error", np.nan)
        h5_evidence = f"corr(uncertainty,error)={corr_ue:.3f}" if np.isfinite(corr_ue) else "corr unavailable"
        h5_conclusion = "supported" if (np.isfinite(corr_ue) and corr_ue > 0) else "weak support"
    rows.append(
        {
            "hypothesis": "H5 inter-model disagreement tracks risk",
            "evidence": h5_evidence,
            "conclusion": h5_conclusion,
        }
    )

    h6_evidence = "CV not run"
    h6_conclusion = "undetermined"
    if cv_summary_df is not None and len(cv_summary_df):
        best_cv = cv_summary_df.sort_values("median_err_mean").iloc[0]
        h6_evidence = (
            f"best={best_cv['model']} mean={best_cv['median_err_mean']:.3f} m, "
            f"std={best_cv['median_err_std']:.3f} over {int(best_cv['n_folds'])} folds"
        )
        h6_conclusion = "supported" if float(best_cv["median_err_std"]) < 2.0 else "partially supported"
    rows.append(
        {
            "hypothesis": "H6 results are robust across grouped splits",
            "evidence": h6_evidence,
            "conclusion": h6_conclusion,
        }
    )

    return pd.DataFrame(rows)


def build_key_summary_lines(
    *,
    cfg: Config,
    seq_bundle: Dict[str, Any] | None = None,
    metrics_protocols: pd.DataFrame | None = None,
    df_eval: pd.DataFrame | None = None,
    comparison_result: Dict[str, Any] | None = None,
    ablation_df: pd.DataFrame | None = None,
    uncertainty_summary: Dict[str, Any] | None = None,
    cross_df: pd.DataFrame | None = None,
    cv_summary_df: pd.DataFrame | None = None,
    cv_meta_df: pd.DataFrame | None = None,
    cv_folds_df: pd.DataFrame | None = None,
    metrics_seq: pd.DataFrame | None = None,
    preds_seq_pointwise: Dict[str, np.ndarray] | None = None,
) -> List[str]:
    """Build compact bullet lines for the automatic result summary section."""

    summary_lines: List[str] = []
    consistency_issues: List[str] = []

    dense = pd.DataFrame()
    seq_dense = pd.DataFrame()
    if metrics_protocols is not None and len(metrics_protocols):
        dense = metrics_protocols[metrics_protocols["protocol"] == "dense_aligned"]
        seq_dense = dense[dense["model_family"] == "sequential_dl"].sort_values("median_err_m")
        if len(seq_dense):
            best_dense = seq_dense.iloc[0]
            summary_lines.append(
                f"Best sequential model (dense): {best_dense['model']} with median {best_dense['median_err_m']:.3f} m and p90 {best_dense['p90_err_m']:.3f} m."
            )
            rollout_dense = dense[dense["model"] == "baseline_constant_velocity_rollout"]
            if len(rollout_dense):
                delta = float(best_dense["median_err_m"] - rollout_dense.iloc[0]["median_err_m"])
                summary_lines.append(f"Dense protocol: best seq vs deployable rollout delta median = {delta:.3f} m.")

        for fam in ["LSTM", "GRU"]:
            base_name = f"{fam}_FE"
            kf_name = f"{fam}_FE_KF"
            d_fam = dense[dense["model"].isin([base_name, kf_name])]
            if len(d_fam) == 2:
                d = d_fam.set_index("model")
                dk = float(d.loc[kf_name, "median_err_m"] - d.loc[base_name, "median_err_m"])
                summary_lines.append(f"{fam} KF delta (KF-base): {dk:.3f} m.")

    if isinstance(seq_bundle, dict):
        summary_lines.append(
            f"Sequence coverage: {seq_bundle['seq_coverage']:.3f} with {seq_bundle['n_test_windows']} windows over {seq_bundle['n_total_test']} evaluation rows."
        )
        diag = seq_bundle.get("seq_diagnostics", {})
        if diag:
            summary_lines.append(
                f"Temporal setup: ignore_time_checks={diag.get('ignore_time_checks', np.nan)}, densify_step_ms={diag.get('densify_step_ms', np.nan)}, test_densify_factor={diag.get('test_densify_factor', np.nan):.2f}."
            )

    if df_eval is not None and len(df_eval):
        d_tmp = df_eval.sort_values(["session_id", "segment_id", cfg.time_col], kind="mergesort").copy()
        dx = d_tmp.groupby(["session_id", "segment_id"])["label_X"].diff()
        dy = d_tmp.groupby(["session_id", "segment_id"])["label_Y"].diff()
        step = np.sqrt(dx**2 + dy**2)
        zero_ratio = float((step.fillna(0.0) <= 1e-9).mean())
        summary_lines.append(f"Dense-label zero-step ratio: {zero_ratio:.3f}.")

    if comparison_result is not None:
        summary_lines.append(
            f"Bootstrap ({comparison_result.get('protocol', 'dense_aligned')}): delta seq-rollout = {comparison_result['delta_median_seq_minus_rollout']:.3f} m "
            f"(95% CI [{comparison_result['ci95_low']:.3f}, {comparison_result['ci95_high']:.3f}])."
        )

    if ablation_df is not None and len(ablation_df):
        ok = ablation_df[ablation_df["status"] == "ok"].copy()
        if len(ok):
            best_abl = ok.sort_values("best_median_err_m").iloc[0]
            worst_abl = ok.sort_values("best_median_err_m").iloc[-1]
            summary_lines.append(
                f"Ablation spread: best {best_abl['tag']}={best_abl['best_median_err_m']:.3f} m vs worst {worst_abl['tag']}={worst_abl['best_median_err_m']:.3f} m."
            )
            if {"base", "strict_time"}.issubset(set(ok["tag"])):
                t = ok.set_index("tag")
                delta = float(t.loc["strict_time", "best_median_err_m"] - t.loc["base", "best_median_err_m"])
                msg = f"Strict-time ablation delta (strict_time - base): {delta:.3f} m."
                if abs(delta) < 1e-3:
                    msg += " (effectively identical in this run)"
                summary_lines.append(msg)
            if {"base", "no_densify"}.issubset(set(ok["tag"])):
                t = ok.set_index("tag")
                n_base = int(t.loc["base", "n_test_windows"])
                n_nd = int(t.loc["no_densify", "n_test_windows"])
                summary_lines.append(
                    f"No-densify comparison caveat: {n_nd} windows vs {n_base} in base (support not strictly matched)."
                )
            if {"base", "topk_off"}.issubset(set(ok["tag"])):
                k = ok.set_index("tag")
                delta = float(k.loc["topk_off", "best_median_err_m"] - k.loc["base", "best_median_err_m"])
                dfeat = float(k.loc["topk_off", "n_features"] - k.loc["base", "n_features"])
                summary_lines.append(f"Top-k toggle (topk_off - base): {delta:.3f} m; feature delta: {dfeat:+.0f}.")
            if {"base", "topk_k10"}.issubset(set(ok["tag"])):
                k = ok.set_index("tag")
                delta = float(k.loc["topk_k10", "best_median_err_m"] - k.loc["base", "best_median_err_m"])
                dfeat = float(k.loc["topk_k10", "n_features"] - k.loc["base", "n_features"])
                summary_lines.append(f"Top-k (k=10) delta vs base: {delta:.3f} m; feature delta: {dfeat:+.0f}.")
            if {"pca_off", "pca_on"}.issubset(set(ok["tag"])):
                pca_cmp = ok.set_index("tag")
                delta = float(pca_cmp.loc["pca_on", "best_median_err_m"] - pca_cmp.loc["pca_off", "best_median_err_m"])
                summary_lines.append(f"PCA toggle (pca_on - pca_off): {delta:.3f} m.")

    if uncertainty_summary is not None and len(uncertainty_summary):
        corr_ue = uncertainty_summary.get("corr_uncertainty_error", np.nan)
        if np.isfinite(corr_ue):
            summary_lines.append(f"Uncertainty/error correlation: {corr_ue:.3f}.")

    if cross_df is not None and len(cross_df):
        cross_seq = cross_df[cross_df["model"].isin(SEQ_MODELS)]
        if len(cross_seq):
            cross_med = float(cross_seq["median_err_m"].median())
            summary_lines.append(f"Cross-device seq median error: {cross_med:.3f} m.")
            if len(seq_dense):
                intra_med = float(seq_dense["median_err_m"].median())
                if intra_med > 0:
                    summary_lines.append(f"Cross/intra median ratio (seq): {cross_med / intra_med:.2f}x.")
            by_pair = cross_seq.groupby("pair", as_index=False)["median_err_m"].median().sort_values("median_err_m")
            if len(by_pair):
                p_best = by_pair.iloc[0]
                p_worst = by_pair.iloc[-1]
                summary_lines.append(
                    f"Directional asymmetry: best {p_best['pair']}={p_best['median_err_m']:.3f} m, worst {p_worst['pair']}={p_worst['median_err_m']:.3f} m."
                )
            tail_ratio = np.nanmedian(cross_seq["p90_err_m"].to_numpy() / np.clip(cross_seq["median_err_m"].to_numpy(), 1e-9, None))
            if np.isfinite(tail_ratio):
                summary_lines.append(f"Cross-device tail ratio median (p90/median): {tail_ratio:.2f}x.")

    if cv_summary_df is not None and len(cv_summary_df):
        best_cv = cv_summary_df.sort_values("median_err_mean").iloc[0]
        summary_lines.append(
            f"CV robustness: best {best_cv['model']} mean={best_cv['median_err_mean']:.3f} m, std={best_cv['median_err_std']:.3f} over {int(best_cv['n_folds'])} folds."
        )
    if cv_meta_df is not None and len(cv_meta_df) and "test_sessions" in cv_meta_df.columns:
        summary_lines.append(
            f"Grouped CV test sessions per split: median={float(cv_meta_df['test_sessions'].median()):.1f}, min={int(cv_meta_df['test_sessions'].min())}, max={int(cv_meta_df['test_sessions'].max())}."
        )
    if cv_folds_df is not None and len(cv_folds_df):
        seq_folds = cv_folds_df[cv_folds_df["model"].isin(SEQ_MODELS)]
        if len(seq_folds):
            fold_med = seq_folds.groupby("fold")["median_err_m"].median().sort_values()
            if len(fold_med):
                summary_lines.append(f"CV fold spread (seq median error): best={fold_med.iloc[0]:.3f} m, worst={fold_med.iloc[-1]:.3f} m.")

    if metrics_seq is not None and len(metrics_seq):
        q_cols = ["median_err_m", "p68_err_m", "p90_err_m", "p95_err_m", "p99_err_m", "max_err_m"]
        present_q = [c for c in q_cols if c in metrics_seq.columns]
        if len(present_q) >= 2:
            vals = metrics_seq[present_q].to_numpy(dtype=float)
            bad_order = np.where(np.any(np.diff(vals, axis=1) < -1e-9, axis=1))[0]
            if len(bad_order):
                bad_models = metrics_seq.iloc[bad_order]["model"].tolist()
                consistency_issues.append(f"non-monotonic quantiles for models: {bad_models}")
        for c in ["median_err_m", "p90_err_m", "rmse_2d_m"]:
            if c in metrics_seq.columns and not np.isfinite(metrics_seq[c]).all():
                consistency_issues.append(f"non-finite values detected in metrics_seq column: {c}")
    else:
        consistency_issues.append("metrics_seq table unavailable")

    if isinstance(preds_seq_pointwise, dict):
        expected_n = None
        if isinstance(seq_bundle, dict):
            expected_n = int(seq_bundle.get("n_test_windows", -1))
        for model_name, pred in preds_seq_pointwise.items():
            arr = np.asarray(pred)
            if arr.ndim != 2 or arr.shape[1] != 2:
                consistency_issues.append(f"{model_name}: unexpected prediction shape {arr.shape}")
                continue
            n_fin = int(np.isfinite(arr).all(axis=1).sum())
            if expected_n is not None and expected_n >= 0 and n_fin != expected_n:
                consistency_issues.append(f"{model_name}: finite predictions {n_fin} != expected test windows {expected_n}")
    else:
        consistency_issues.append("preds_seq_pointwise unavailable")

    if consistency_issues:
        summary_lines.append("Consistency checks: warnings detected.")
        for msg in consistency_issues[:4]:
            summary_lines.append(f"Consistency warning: {msg}.")
    else:
        summary_lines.append("Consistency checks: no blocking inconsistency found in metrics tables or prediction support.")

    return summary_lines


def build_conclusion_markdown(
    *,
    cfg: Config,
    seq_bundle: Dict[str, Any] | None = None,
    metrics_protocols: pd.DataFrame | None = None,
    ablation_df: pd.DataFrame | None = None,
    cross_df: pd.DataFrame | None = None,
    cv_summary_df: pd.DataFrame | None = None,
    cv_folds_df: pd.DataFrame | None = None,
    uncertainty_summary: Dict[str, Any] | None = None,
    df_eval: pd.DataFrame | None = None,
    seg_stats: pd.DataFrame | None = None,
    metrics_seq: pd.DataFrame | None = None,
    preds_seq_pointwise: Dict[str, np.ndarray] | None = None,
) -> str:
    """Build the final general conclusion markdown block."""

    def _fmt(x: float | int | None, nd: int = 3) -> str:
        try:
            if x is None or not np.isfinite(float(x)):
                return "n/a"
            return f"{float(x):.{nd}f}"
        except Exception:
            return "n/a"

    def _fmt_pct(x: float | int | None, nd: int = 1) -> str:
        try:
            if x is None or not np.isfinite(float(x)):
                return "n/a"
            return f"{float(x) * 100:.{nd}f}%"
        except Exception:
            return "n/a"

    # Base quantitative anchors
    best_model = None
    best_med = None
    best_p90 = None
    best_p1 = None
    best_p2 = None
    rollout_med = None
    delta_vs_rollout = None
    tie_msg = ""

    if metrics_protocols is not None and len(metrics_protocols):
        dense = metrics_protocols[metrics_protocols["protocol"] == "dense_aligned"]
        seq_dense = dense[dense["model_family"] == "sequential_dl"].sort_values("median_err_m")
        if len(seq_dense):
            bd = seq_dense.iloc[0]
            best_model = str(bd["model"])
            best_med = float(bd["median_err_m"])
            best_p90 = float(bd["p90_err_m"])
            best_p1 = bd.get("p_err_lt_1.0m")
            best_p2 = bd.get("p_err_lt_2.0m")
            if len(seq_dense) > 1:
                sd = seq_dense.iloc[1]
                gap = float(sd["median_err_m"] - best_med)
                if np.isfinite(gap) and gap <= 0.1:
                    tie_msg = (
                        f"{best_model} and {sd['model']} are essentially tied "
                        f"(median gap {_fmt(gap)} m)."
                    )
        rollout = dense[dense["model"] == "baseline_constant_velocity_rollout"]
        if len(rollout):
            rollout_med = float(rollout.iloc[0]["median_err_m"])
        if (best_med is not None) and (rollout_med is not None):
            delta_vs_rollout = float(best_med - rollout_med)

    # Protocol / coverage context
    coverage_msg = ""
    densify_msg = ""
    if isinstance(seq_bundle, dict):
        if "seq_coverage" in seq_bundle and "n_test_windows" in seq_bundle and "n_total_test" in seq_bundle:
            coverage_msg = (
                f"Coverage={_fmt(seq_bundle['seq_coverage'])} "
                f"({int(seq_bundle['n_test_windows'])}/{int(seq_bundle['n_total_test'])} windows)."
            )
        if "densify_step_ms" in seq_bundle and "test_densify_factor" in seq_bundle:
            densify_msg = (
                f"Densify step={seq_bundle.get('densify_step_ms', 'n/a')} ms, "
                f"test densify factor={_fmt(seq_bundle.get('test_densify_factor'))}."
            )

    # Label staircase
    staircase_msg = ""
    if df_eval is not None and len(df_eval):
        d_tmp = df_eval.sort_values(["session_id", "segment_id", cfg.time_col], kind="mergesort").copy()
        dx = d_tmp.groupby(["session_id", "segment_id"])["label_X"].diff()
        dy = d_tmp.groupby(["session_id", "segment_id"])["label_Y"].diff()
        step = np.sqrt(dx**2 + dy**2)
        zero_ratio = float((step.fillna(0.0) <= 1e-9).mean())
        staircase_msg = (
            f"Dense labels remain staircase-like (zero-step ratio {_fmt(zero_ratio, 3)}), "
            "which keeps dense-aligned baselines optimistic."
        )

    # Ablation signals
    topk_msg = ""
    topk_k_msg = ""
    window_msg = ""
    densify_ablation_msg = ""
    strict_msg = ""
    if ablation_df is not None and len(ablation_df):
        ok = ablation_df[ablation_df["status"] == "ok"].copy()
        if len(ok) and {"base", "topk_off"}.issubset(set(ok["tag"])):
            k = ok.set_index("tag")
            delta = float(k.loc["topk_off", "best_median_err_m"] - k.loc["base", "best_median_err_m"])
            topk_msg = f"Turning off top-k correlation adds {_fmt(delta)} m to the median error."
        if len(ok) and {"base", "topk_k10"}.issubset(set(ok["tag"])):
            k = ok.set_index("tag")
            delta = float(k.loc["topk_k10", "best_median_err_m"] - k.loc["base", "best_median_err_m"])
            topk_k_msg = f"A smaller top-k (k=10) is even worse by {_fmt(delta)} m."
        if len(ok) and {"base", "short_window", "long_window"}.issubset(set(ok["tag"])):
            k = ok.set_index("tag")
            d_short = float(k.loc["short_window", "best_median_err_m"] - k.loc["base", "best_median_err_m"])
            d_long = float(k.loc["long_window", "best_median_err_m"] - k.loc["base", "best_median_err_m"])
            window_msg = (
                f"Window-length changes degrade performance (short: +{_fmt(d_short)} m, "
                f"long: +{_fmt(d_long)} m)."
            )
        if len(ok) and {"base", "no_densify"}.issubset(set(ok["tag"])):
            k = ok.set_index("tag")
            n_base = int(k.loc["base", "n_test_windows"])
            n_nd = int(k.loc["no_densify", "n_test_windows"])
            densify_ablation_msg = (
                f"No-densify runs on fewer windows ({n_nd} vs {n_base}), "
                "so its delta mixes protocol and model effects."
            )
        if len(ok) and {"base", "strict_time"}.issubset(set(ok["tag"])):
            k = ok.set_index("tag")
            delta = float(k.loc["strict_time", "best_median_err_m"] - k.loc["base", "best_median_err_m"])
            if abs(delta) < 1e-3:
                strict_msg = "Strict time checks did not change the support or the metrics in this run."

    # Cross-device behavior
    cross_msg = ""
    cross_tail_msg = ""
    if cross_df is not None and len(cross_df):
        cross_seq = cross_df[cross_df["model"].isin(SEQ_MODELS)]
        if len(cross_seq):
            cross_med = float(cross_seq["median_err_m"].median())
            by_pair = cross_seq.groupby("pair", as_index=False)["median_err_m"].median().sort_values("median_err_m")
            p_best = by_pair.iloc[0]
            p_worst = by_pair.iloc[-1]
            cross_msg = (
                f"Cross-device median error is {cross_med:.3f} m, "
                f"with strong directional asymmetry ({p_best['pair']}={p_best['median_err_m']:.3f} m "
                f"vs {p_worst['pair']}={p_worst['median_err_m']:.3f} m)."
            )
            best_pair_rows = cross_seq[cross_seq["pair"] == p_best["pair"]].sort_values("median_err_m")
            if len(best_pair_rows):
                best_pair = best_pair_rows.iloc[0]
                if np.isfinite(best_pair["p90_err_m"]) and np.isfinite(best_pair["median_err_m"]):
                    tail = float(best_pair["p90_err_m"] / best_pair["median_err_m"])
                    cross_tail_msg = f"Cross-device tails are heavy (best pair p90/median={_fmt(tail)}x)."

    # CV variability
    cv_msg = ""
    cv_spread_msg = ""
    if cv_summary_df is not None and len(cv_summary_df):
        best_cv = cv_summary_df.sort_values("median_err_mean").iloc[0]
        cv_msg = (
            f"Grouped CV remains high-variance: best mean={best_cv['median_err_mean']:.3f} m "
            f"with std={best_cv['median_err_std']:.3f} over {int(best_cv['n_folds'])} folds."
        )
    if cv_folds_df is not None and len(cv_folds_df):
        seq_folds = cv_folds_df[cv_folds_df["model"].isin(SEQ_MODELS)]
        if len(seq_folds):
            fold_med = seq_folds.groupby("fold")["median_err_m"].median().sort_values()
            if len(fold_med):
                cv_spread_msg = (
                    f"Fold medians range from {_fmt(fold_med.iloc[0])} m to {_fmt(fold_med.iloc[-1])} m."
                )

    # Uncertainty utility (if available)
    uncert_msg = ""
    if uncertainty_summary is not None and len(uncertainty_summary):
        corr_ue = uncertainty_summary.get("corr_uncertainty_error", np.nan)
        if np.isfinite(corr_ue):
            uncert_msg = f"Inter-model disagreement is informative (corr={_fmt(corr_ue)})."

    best_msg = ""
    if best_model is not None and best_med is not None:
        best_msg = (
            f"Best sequential model is {best_model} (median {best_med:.3f} m, "
            f"p90 {best_p90:.3f} m, "
            f"P(err<1 m)={_fmt_pct(best_p1)}, P(err<2 m)={_fmt_pct(best_p2)})."
        )
        if delta_vs_rollout is not None:
            best_msg += f" Delta vs rollout baseline: {delta_vs_rollout:.3f} m (median)."

    # Consistency checks (lightweight, report-friendly)
    consistency_msg = ""
    if metrics_seq is not None and len(metrics_seq):
        q_cols = [c for c in ["median_err_m", "p68_err_m", "p90_err_m", "p95_err_m", "p99_err_m", "max_err_m"] if c in metrics_seq.columns]
        if q_cols:
            vals = metrics_seq[q_cols].to_numpy(dtype=float)
            if not np.any(np.diff(vals, axis=1) < -1e-9):
                consistency_msg = "Consistency checks did not reveal blocking issues in the reported metrics."

    worked_lines = [line for line in [best_msg, tie_msg, coverage_msg, densify_msg] if line]
    ablation_lines = [line for line in [topk_msg, topk_k_msg, window_msg, densify_ablation_msg, strict_msg] if line]
    limit_lines = [line for line in [staircase_msg, cross_msg, cross_tail_msg, cv_msg, cv_spread_msg] if line]
    method_lines = [line for line in [uncert_msg, consistency_msg] if line]

    worked_block = "\n".join([f"- {l}" for l in worked_lines]) if worked_lines else ""
    ablation_block = "\n".join([f"- {l}" for l in ablation_lines]) if ablation_lines else ""
    limit_block = "\n".join([f"- {l}" for l in limit_lines]) if limit_lines else ""
    method_block = "\n".join([f"- {l}" for l in method_lines]) if method_lines else ""

    improvements = [
        "tighten temporal protocols (strict checks and label-sparsity-aware evaluation) before further architecture tuning",
        "prioritize cross-device robustness with explicit domain adaptation or device normalization",
        "expand or rebalance sessions to reduce fold variance and stabilize conclusions",
    ]
    improve_block = "\n".join([f"- {l}" for l in improvements])

    base_answer = "The dense protocol shows that within-device trajectory dynamics are recoverable under the current labeling density."
    if best_msg:
        base_answer = f"{base_answer} {best_msg}"

    narrative = (
        "This work builds a full temporal pipeline for indoor localization from WiFi+IMU streams, "
        "centered on LSTM/GRU sequence models with a diamond head and evaluated under leakage-safe, "
        "cross-device-aware protocols."
    )

    sections = []
    if worked_block:
        sections.append(f"**Result synthesis**\n{worked_block}")
    if ablation_block:
        sections.append(f"**Ablations and protocol sensitivity**\n{ablation_block}")
    if limit_block:
        sections.append(f"**Limits and robustness**\n{limit_block}")
    if method_block:
        sections.append(f"**Methodology checks**\n{method_block}")
    sections.append(f"**Most promising improvements**\n{improve_block}")

    return "\n\n".join([narrative, f"**Does it answer the base problem?**  \n{base_answer}"] + sections)
