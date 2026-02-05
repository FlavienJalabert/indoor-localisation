"""Evaluation orchestration with caching.

This module:
- prepares tabular and sequential bundles (fit/transform discipline)
- trains/evaluates models
- caches artifacts under outputs/ via artifacts.py
No plotting here.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import Config
from artifacts import (
    artifact_path,
    exists,
    save_df,
    load_df,
    save_npz,
    load_npz,
    save_joblib,
    load_joblib,
    save_json,
    load_json,
)
from feature_engineering import feature_engineering_best, fit_wifi_selector
from tabular_models import (
    build_preprocessor,
    train_knn,
    train_rf,
    train_xgb_xy,
    predict_xy_from_xgb,
)
from metrics import evaluate_regression
from splitting import add_session_id, group_split_train_val
from seq_training import (
    build_sequences,
    TrajDataset,
    fit_seq_scaler,
    transform_seq_scaler,
    train_torch_model,
    predict_torch_model,
    seq_preds_to_pointwise,
)
from seq_models import LSTMRegressorDiamond, GRURegressorDiamond
from filters import kalman_filter_2d


# -------------------------
# small helpers
# -------------------------

def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _get_cfg(cfg: Config, key: str, default):
    # avoids hard-crash if your Config is slightly different
    return getattr(cfg, key, default)


def _dict_npz_to_preds(d: dict) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in d.items()}


def _reconstruct_positions_from_deltas(
    df_test_base: pd.DataFrame, idx_seq: list[int], y_pred_delta: np.ndarray
) -> np.ndarray:
    """Reconstruct absolute positions from delta predictions, anchored at the first true point per segment."""

    idx_seq = np.asarray(idx_seq, dtype=int)
    if "session_id" not in df_test_base.columns:
        df_test_base = add_session_id(df_test_base)

    group_cols = ["session_id"]
    if "segment_id" in df_test_base.columns:
        group_cols.append("segment_id")

    seq_meta = df_test_base.loc[idx_seq, group_cols + ["label_X", "label_Y"]].copy()
    seq_meta["seq_pos"] = np.arange(len(idx_seq))

    y_pred_pos = np.zeros((len(idx_seq), 2), dtype=float)

    for _, g in seq_meta.groupby(group_cols):
        # preserve sequence order as built in build_sequences (seq_pos), not original index
        g_sorted = g.sort_values("seq_pos")
        seq_idx = g_sorted["seq_pos"].to_numpy()
        start = g_sorted[["label_X", "label_Y"]].iloc[0].to_numpy(dtype=float)
        deltas = y_pred_delta[seq_idx]
        pos = np.zeros_like(deltas)
        pos[0] = start
        if len(deltas) > 1:
            pos[1:] = start + np.cumsum(deltas[1:], axis=0)
        y_pred_pos[seq_idx] = pos

    return y_pred_pos


def baseline_last_position(df: pd.DataFrame, *, time_col: str = "t_ms") -> np.ndarray:
    """Predict previous position within each session (persistence baseline)."""

    df_work = df.copy().reset_index(drop=True)
    if "session_id" not in df_work.columns:
        df_work = add_session_id(df_work)
    df_work["__row_id"] = np.arange(len(df_work))

    group_cols = ["session_id"]
    if "segment_id" in df_work.columns:
        group_cols.append("segment_id")

    out = np.zeros((len(df_work), 2), dtype=float)
    for _, g in df_work.groupby(group_cols, sort=False):
        if time_col in g.columns:
            g_sorted = g.sort_values(time_col, kind="mergesort")
        else:
            g_sorted = g.sort_values("__row_id", kind="mergesort")
        y = g_sorted[["label_X", "label_Y"]].to_numpy(dtype=float)
        if len(y) == 0:
            continue
        pred = np.vstack([y[:1], y[:-1]])
        out[g_sorted["__row_id"].to_numpy(dtype=int)] = pred
    return out


def baseline_constant_velocity(
    df: pd.DataFrame,
    *,
    time_col: str = "t_ms",
    max_speed_mps: float | None = 2.5,
    max_dt_ms: float | None = 1000.0,
) -> np.ndarray:
    """Predict next position using constant velocity from last two points."""

    df_work = df.copy().reset_index(drop=True)
    if "session_id" not in df_work.columns:
        df_work = add_session_id(df_work)
    df_work["__row_id"] = np.arange(len(df_work))

    group_cols = ["session_id"]
    if "segment_id" in df_work.columns:
        group_cols.append("segment_id")

    out = np.zeros((len(df_work), 2), dtype=float)
    for _, g in df_work.groupby(group_cols, sort=False):
        if time_col in g.columns:
            g_sorted = g.sort_values(time_col, kind="mergesort")
            t = g_sorted[time_col].to_numpy(dtype=float)
        else:
            g_sorted = g.sort_values("__row_id", kind="mergesort")
            t = np.arange(len(g_sorted), dtype=float)
        y = g_sorted[["label_X", "label_Y"]].to_numpy(dtype=float)
        if len(y) == 0:
            continue
        pred = np.zeros_like(y)
        pred[0] = y[0]
        if len(y) > 1:
            pred[1] = y[0]
        for i in range(2, len(y)):
            dt_prev = (t[i - 1] - t[i - 2]) / 1000.0
            dt_curr = (t[i] - t[i - 1]) / 1000.0
            if (
                not np.isfinite(dt_prev)
                or not np.isfinite(dt_curr)
                or dt_prev <= 0
                or dt_curr <= 0
                or (max_dt_ms is not None and ((dt_prev * 1000.0 > max_dt_ms) or (dt_curr * 1000.0 > max_dt_ms)))
            ):
                pred[i] = y[i - 1]
                continue
            v = (y[i - 1] - y[i - 2]) / dt_prev
            if max_speed_mps is not None and max_speed_mps > 0:
                speed = float(np.linalg.norm(v))
                if np.isfinite(speed) and speed > max_speed_mps:
                    v = v * (max_speed_mps / max(speed, 1e-12))
            pred[i] = y[i - 1] + v * dt_curr
        out[g_sorted["__row_id"].to_numpy(dtype=int)] = pred
    return out


def baseline_constant_velocity_rollout(
    df: pd.DataFrame,
    *,
    time_col: str = "t_ms",
    max_speed_mps: float | None = 2.5,
    max_dt_ms: float | None = 1000.0,
) -> np.ndarray:
    """Autoregressive constant-velocity baseline (uses previous predictions, not true labels)."""

    df_work = df.copy().reset_index(drop=True)
    if "session_id" not in df_work.columns:
        df_work = add_session_id(df_work)
    df_work["__row_id"] = np.arange(len(df_work))

    group_cols = ["session_id"]
    if "segment_id" in df_work.columns:
        group_cols.append("segment_id")

    out = np.zeros((len(df_work), 2), dtype=float)
    for _, g in df_work.groupby(group_cols, sort=False):
        if time_col in g.columns:
            g_sorted = g.sort_values(time_col, kind="mergesort")
            t = g_sorted[time_col].to_numpy(dtype=float)
        else:
            g_sorted = g.sort_values("__row_id", kind="mergesort")
            t = np.arange(len(g_sorted), dtype=float)
        y = g_sorted[["label_X", "label_Y"]].to_numpy(dtype=float)
        if len(y) == 0:
            continue
        pred = np.zeros_like(y)
        pred[0] = y[0]
        if len(y) > 1:
            pred[1] = y[1]
        for i in range(2, len(y)):
            dt_prev = (t[i - 1] - t[i - 2]) / 1000.0
            dt_curr = (t[i] - t[i - 1]) / 1000.0
            if (
                not np.isfinite(dt_prev)
                or not np.isfinite(dt_curr)
                or dt_prev <= 0
                or dt_curr <= 0
                or (max_dt_ms is not None and ((dt_prev * 1000.0 > max_dt_ms) or (dt_curr * 1000.0 > max_dt_ms)))
            ):
                pred[i] = pred[i - 1]
                continue
            v = (pred[i - 1] - pred[i - 2]) / dt_prev
            if max_speed_mps is not None and max_speed_mps > 0:
                speed = float(np.linalg.norm(v))
                if np.isfinite(speed) and speed > max_speed_mps:
                    v = v * (max_speed_mps / max(speed, 1e-12))
            pred[i] = pred[i - 1] + v * dt_curr
        out[g_sorted["__row_id"].to_numpy(dtype=int)] = pred
    return out


def _apply_kalman_by_group(
    y_pred: np.ndarray,
    df_ref: pd.DataFrame,
    *,
    time_col: str = "t_ms",
    process_var: float = 1e-3,
    meas_var: float = 1e-1,
) -> np.ndarray:
    """Apply Kalman smoothing independently on each session/segment group."""

    y_pred = np.asarray(y_pred, dtype=float)
    if y_pred.ndim != 2 or y_pred.shape[1] != 2:
        raise ValueError("y_pred must be (N,2)")
    if len(df_ref) != y_pred.shape[0]:
        raise ValueError("df_ref and y_pred must have the same length")

    df = df_ref.copy().reset_index(drop=True)
    if "session_id" not in df.columns:
        df = add_session_id(df)
    group_cols = ["session_id"]
    if "segment_id" in df.columns:
        group_cols.append("segment_id")
    df["__row_id"] = np.arange(len(df))

    out = np.zeros_like(y_pred, dtype=float)
    for _, g in df.groupby(group_cols, sort=False):
        if time_col in g.columns:
            g_sorted = g.sort_values(time_col, kind="mergesort")
            t_ms = g_sorted[time_col].to_numpy(dtype=float)
        else:
            g_sorted = g.sort_values("__row_id", kind="mergesort")
            t_ms = None
        idx = g_sorted["__row_id"].to_numpy(dtype=int)
        filt = kalman_filter_2d(
            y_pred[idx],
            t_ms=t_ms,
            process_var=process_var,
            meas_var=meas_var,
        )
        out[idx] = filt
    return out


def _stabilize_predictions_by_group(
    y_pred: np.ndarray,
    df_ref: pd.DataFrame,
    *,
    time_col: str = "t_ms",
    max_speed_mps: float | None = 2.5,
    max_dt_ms: float | None = 1000.0,
) -> np.ndarray:
    """Apply simple kinematic guardrails per session/segment on predicted trajectories."""

    y_pred = np.asarray(y_pred, dtype=float)
    if y_pred.ndim != 2 or y_pred.shape[1] != 2:
        raise ValueError("y_pred must be (N,2)")
    if len(df_ref) != y_pred.shape[0]:
        raise ValueError("df_ref and y_pred must have the same length")

    df = df_ref.copy().reset_index(drop=True)
    if "session_id" not in df.columns:
        df = add_session_id(df)
    df["__row_id"] = np.arange(len(df))

    group_cols = ["session_id"]
    if "segment_id" in df.columns:
        group_cols.append("segment_id")

    out = np.asarray(y_pred, dtype=float).copy()
    for _, g in df.groupby(group_cols, sort=False):
        if time_col in g.columns:
            g_sorted = g.sort_values(time_col, kind="mergesort")
            t = g_sorted[time_col].to_numpy(dtype=float)
        else:
            g_sorted = g.sort_values("__row_id", kind="mergesort")
            t = np.arange(len(g_sorted), dtype=float)
        idx = g_sorted["__row_id"].to_numpy(dtype=int)
        if idx.size <= 1:
            continue

        yp = out[idx].copy()
        for i in range(1, len(idx)):
            dt_s = (t[i] - t[i - 1]) / 1000.0
            if (not np.isfinite(dt_s)) or dt_s <= 0 or (max_dt_ms is not None and dt_s * 1000.0 > max_dt_ms):
                yp[i] = yp[i - 1]
                continue
            step = yp[i] - yp[i - 1]
            step_norm = float(np.linalg.norm(step))
            if not np.isfinite(step_norm):
                yp[i] = yp[i - 1]
                continue
            if max_speed_mps is not None and max_speed_mps > 0:
                max_step = max_speed_mps * dt_s
                if step_norm > max_step:
                    yp[i] = yp[i - 1] + step * (max_step / max(step_norm, 1e-12))

        out[idx] = yp
    return out


# -------------------------
# TABULAR
# -------------------------

def prepare_tabular_xy(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    cfg: Config,
    run_id: str,
    force_recompute: bool,
) -> Dict:
    """Prepare tabular features/labels with caching.

    Returns dict:
        X_train_fe: pd.DataFrame
        X_test_fe:  pd.DataFrame (columns aligned to train)
        y_train: np.ndarray (N,2)
        y_test:  np.ndarray (N,2)
        preprocessor: sklearn ColumnTransformer (not fitted)
        wifi_cols: list[str]
        feature_cols: list[str] (X columns, in order)
        df_test_fe: pd.DataFrame (for later diagnostics, aligned with X_test rows)
    """
    # ---- paths
    p_wifi = artifact_path("fe", "wifi_cols_fixed", run_id, "json")
    p_cols = artifact_path("fe", "feature_cols_train", run_id, "json")

    p_Xtr = artifact_path("fe", "X_train_fe", run_id, "csv")
    p_Xte = artifact_path("fe", "X_test_fe", run_id, "csv")

    p_ytr = artifact_path("fe", "y_train", run_id, "npz")
    p_yte = artifact_path("fe", "y_test", run_id, "npz")

    p_dfte = artifact_path("fe", "df_test_fe", run_id, "csv")
    p_meta = artifact_path("fe", "tabular_meta", run_id, "json")

    if (not force_recompute) and all(map(exists, [p_wifi, p_cols, p_Xtr, p_Xte, p_ytr, p_yte, p_dfte, p_meta])):
        wifi_cols = load_json(p_wifi)["wifi_cols"]
        feature_cols = load_json(p_cols)["feature_cols"]

        X_train_fe = load_df(p_Xtr)
        X_test_fe = load_df(p_Xte)

        y_train = load_npz(p_ytr)["y"]
        y_test = load_npz(p_yte)["y"]

        df_test_fe = load_df(p_dfte)
        meta = load_json(p_meta)
    else:
        # sanity required columns
        _require_cols(df_train, ["device", "motion", "label_X", "label_Y"], "df_train")
        _require_cols(df_test, ["device", "motion", "label_X", "label_Y"], "df_test")

        # --- fit wifi selector on TRAIN only
        wifi_cols = fit_wifi_selector(
            df_train,
            prefixes=tuple(_get_cfg(cfg, "wifi_prefixes", ())),
            rssi_missing=float(_get_cfg(cfg, "rssi_missing", -100.0)),
            wifi_min_presence=float(_get_cfg(cfg, "wifi_min_presence", 0.05)),
            wifi_topk=int(_get_cfg(cfg, "wifi_topk", 10)),
        )
        save_json({"wifi_cols": wifi_cols}, p_wifi)

        # --- FE train
        df_train_fe, X_train_num, _, meta_tr, _ = feature_engineering_best(
            df_train,
            time_col=str(_get_cfg(cfg, "time_col", "t_ms")),
            imu_cols=tuple(_get_cfg(cfg, "imu_cols", ())),
            rolling_window_size=int(_get_cfg(cfg, "rolling_window_size", 10)),
            rolling_group_cols=tuple(_get_cfg(cfg, "rolling_group_cols", ("device", "motion"))),
            rolling_imu_cols=tuple(_get_cfg(cfg, "rolling_imu_cols", ())),
            rolling_stats=tuple(_get_cfg(cfg, "rolling_stats", ("mean", "var", "min", "max"))),
            add_rolling=bool(_get_cfg(cfg, "add_rolling", True)),
            add_diff=bool(_get_cfg(cfg, "add_diff", True)),
            add_dt_derivative=bool(_get_cfg(cfg, "add_dt_derivative", True)),
            eps_dt_ms=float(_get_cfg(cfg, "eps_dt_ms", 1.0)),
            wifi_cols_fixed=wifi_cols,
            wifi_prefixes=tuple(_get_cfg(cfg, "wifi_prefixes", ())),
            rssi_missing=float(_get_cfg(cfg, "rssi_missing", -100.0)),
            fill_numeric_with=str(_get_cfg(cfg, "fill_numeric_with", "median")),
            verbose=False,
        )

        # --- FE test
        df_test_fe, X_test_num, _, meta_te, _ = feature_engineering_best(
            df_test,
            time_col=str(_get_cfg(cfg, "time_col", "t_ms")),
            imu_cols=tuple(_get_cfg(cfg, "imu_cols", ())),
            rolling_window_size=int(_get_cfg(cfg, "rolling_window_size", 10)),
            rolling_group_cols=tuple(_get_cfg(cfg, "rolling_group_cols", ("device", "motion"))),
            rolling_imu_cols=tuple(_get_cfg(cfg, "rolling_imu_cols", ())),
            rolling_stats=tuple(_get_cfg(cfg, "rolling_stats", ("mean", "var", "min", "max"))),
            add_rolling=bool(_get_cfg(cfg, "add_rolling", True)),
            add_diff=bool(_get_cfg(cfg, "add_diff", True)),
            add_dt_derivative=bool(_get_cfg(cfg, "add_dt_derivative", True)),
            eps_dt_ms=float(_get_cfg(cfg, "eps_dt_ms", 1.0)),
            wifi_cols_fixed=wifi_cols,
            wifi_prefixes=tuple(_get_cfg(cfg, "wifi_prefixes", ())),
            rssi_missing=float(_get_cfg(cfg, "rssi_missing", -100.0)),
            fill_numeric_with=str(_get_cfg(cfg, "fill_numeric_with", "median")),
            verbose=False,
        )

        # labels
        y_train = df_train_fe[["label_X", "label_Y"]].to_numpy(dtype=float)
        y_test = df_test_fe[["label_X", "label_Y"]].to_numpy(dtype=float)

        # X = numeric FE + cat
        X_train_fe = pd.concat(
            [X_train_num.reset_index(drop=True), df_train_fe[["device", "motion"]].reset_index(drop=True)],
            axis=1,
        )
        X_test_fe = pd.concat(
            [X_test_num.reset_index(drop=True), df_test_fe[["device", "motion"]].reset_index(drop=True)],
            axis=1,
        )

        # strict alignment
        X_test_fe = X_test_fe.reindex(columns=X_train_fe.columns, fill_value=0.0)

        feature_cols = list(X_train_fe.columns)
        save_json({"feature_cols": feature_cols}, p_cols)
        save_df(X_train_fe, p_Xtr)
        save_df(X_test_fe, p_Xte)

        save_npz(p_ytr, y=y_train)
        save_npz(p_yte, y=y_test)

        save_df(df_test_fe.reset_index(drop=True), p_dfte)

        meta = {
            "wifi_cols": wifi_cols,
            "X_train_shape": list(X_train_fe.shape),
            "X_test_shape": list(X_test_fe.shape),
            "meta_train_fe": meta_tr,
            "meta_test_fe": meta_te,
        }
        save_json(meta, p_meta)

    # preprocessor (not fitted here)
    cat_cols = ["device", "motion"]
    num_cols = [c for c in X_train_fe.columns if c not in cat_cols]
    preprocessor = build_preprocessor(num_cols, cat_cols)

    return {
        "X_train_fe": X_train_fe,
        "X_test_fe": X_test_fe,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "wifi_cols": wifi_cols,
        "feature_cols": list(X_train_fe.columns),
        "df_test_fe": df_test_fe,
        "meta": meta,
    }


def eval_tabular_models(bundle: Dict, *, cfg: Config) -> Tuple[pd.DataFrame, Dict]:
    """Train/evaluate tabular models and return metrics and predictions.

    NOTE: caching uses cfg.run_id if present? We don't have run_id in signature,
    so we rely on cfg to expose `run_id` OR you handle caching in notebook by calling
    this only when needed.

    Recommended: add run_id + force_recompute to signature later.
    For now: no caching here (or minimal caching with cfg.run_id if present).
    """
    # You chose signature without run_id/force_recompute; we still can cache via cfg if present.
    run_id = getattr(cfg, "run_id", None)
    force_recompute = bool(getattr(cfg, "force_recompute", False))

    X_train = bundle["X_train_fe"]
    X_test = bundle["X_test_fe"]
    y_train = bundle["y_train"]
    y_test = bundle["y_test"]
    preproc = bundle["preprocessor"]
    df_test_fe = bundle.get("df_test_fe")

    # Optional caching
    if run_id is not None:
        p_results_pkl = artifact_path("metrics", "results_tabular", run_id, "joblib")
        p_results_csv = artifact_path("metrics", "results_tabular", run_id, "csv")
        p_preds = artifact_path("metrics", "preds_tabular", run_id, "npz")
        if (not force_recompute) and all(map(exists, [p_results_pkl, p_preds])):
            results_df = load_joblib(p_results_pkl)
            preds_pack = load_npz(p_preds)
            preds_by_model = _dict_npz_to_preds(preds_pack)
            return results_df, preds_by_model
    else:
        p_results_pkl = p_results_csv = p_preds = None

    thresholds = tuple(_get_cfg(cfg, "thresholds", (0.25, 0.5, 1.0, 2.0)))
    use_kalman = bool(_get_cfg(cfg, "use_kalman_postproc", True))
    use_guardrails = bool(_get_cfg(cfg, "use_pred_guardrails", True))
    time_col = str(_get_cfg(cfg, "time_col", "t_ms"))
    max_speed_mps = _get_cfg(cfg, "baseline_max_speed_mps", 2.5)
    max_speed_mps = float(max_speed_mps) if max_speed_mps is not None else None
    max_dt_ms = _get_cfg(cfg, "gap_thr_ms", 1000.0)
    max_dt_ms = float(max_dt_ms) if max_dt_ms is not None else None
    kalman_process_var = float(_get_cfg(cfg, "kalman_process_var", 1e-3))
    kalman_meas_var = float(_get_cfg(cfg, "kalman_meas_var", 1e-1))

    preds_by_model: Dict[str, np.ndarray] = {}
    rows = []

    # kNN
    knn = train_knn(X_train, y_train, preproc, knn_params=dict(_get_cfg(cfg, "knn_params", {})))
    y_pred = knn.predict(X_test)
    if use_guardrails and df_test_fe is not None:
        y_pred = _stabilize_predictions_by_group(
            y_pred, df_test_fe, time_col=time_col, max_speed_mps=max_speed_mps, max_dt_ms=max_dt_ms
        )
    preds_by_model["kNN_FE"] = np.asarray(y_pred, dtype=float)
    rows.append(evaluate_regression(y_test, y_pred, "kNN_FE", thresholds=thresholds))
    if use_kalman and df_test_fe is not None:
        y_pred_kf = _apply_kalman_by_group(
            y_pred, df_test_fe, time_col=time_col, process_var=kalman_process_var, meas_var=kalman_meas_var
        )
        preds_by_model["kNN_FE_KF"] = np.asarray(y_pred_kf, dtype=float)
        rows.append(evaluate_regression(y_test, y_pred_kf, "kNN_FE_KF", thresholds=thresholds))

    # RF
    rf = train_rf(X_train, y_train, preproc, rf_params=dict(_get_cfg(cfg, "rf_params", {})))
    y_pred = rf.predict(X_test)
    if use_guardrails and df_test_fe is not None:
        y_pred = _stabilize_predictions_by_group(
            y_pred, df_test_fe, time_col=time_col, max_speed_mps=max_speed_mps, max_dt_ms=max_dt_ms
        )
    preds_by_model["RandomForest_FE"] = np.asarray(y_pred, dtype=float)
    rows.append(evaluate_regression(y_test, y_pred, "RandomForest_FE", thresholds=thresholds))
    if use_kalman and df_test_fe is not None:
        y_pred_kf = _apply_kalman_by_group(
            y_pred, df_test_fe, time_col=time_col, process_var=kalman_process_var, meas_var=kalman_meas_var
        )
        preds_by_model["RandomForest_FE_KF"] = np.asarray(y_pred_kf, dtype=float)
        rows.append(evaluate_regression(y_test, y_pred_kf, "RandomForest_FE_KF", thresholds=thresholds))

    # XGB (x,y)
    xgb_x, xgb_y = train_xgb_xy(X_train, y_train, preproc, xgb_params=dict(_get_cfg(cfg, "xgb_params", {})))
    y_pred = predict_xy_from_xgb(xgb_x, xgb_y, X_test)
    if use_guardrails and df_test_fe is not None:
        y_pred = _stabilize_predictions_by_group(
            y_pred, df_test_fe, time_col=time_col, max_speed_mps=max_speed_mps, max_dt_ms=max_dt_ms
        )
    preds_by_model["XGBoost_FE"] = np.asarray(y_pred, dtype=float)
    rows.append(evaluate_regression(y_test, y_pred, "XGBoost_FE", thresholds=thresholds))
    if use_kalman and df_test_fe is not None:
        y_pred_kf = _apply_kalman_by_group(
            y_pred, df_test_fe, time_col=time_col, process_var=kalman_process_var, meas_var=kalman_meas_var
        )
        preds_by_model["XGBoost_FE_KF"] = np.asarray(y_pred_kf, dtype=float)
        rows.append(evaluate_regression(y_test, y_pred_kf, "XGBoost_FE_KF", thresholds=thresholds))

    results_df = pd.DataFrame(rows).sort_values("median_err_m").reset_index(drop=True)

    # save caches (if run_id provided)
    if run_id is not None:
        save_joblib(knn, artifact_path("models", "knn_tabular", run_id, "joblib"))
        save_joblib(rf, artifact_path("models", "rf_tabular", run_id, "joblib"))
        save_joblib(xgb_x, artifact_path("models", "xgb_x", run_id, "joblib"))
        save_joblib(xgb_y, artifact_path("models", "xgb_y", run_id, "joblib"))

        results_df.to_csv(p_results_csv, index=False)
        save_joblib(results_df, p_results_pkl)
        save_npz(p_preds, **{k: v for k, v in preds_by_model.items()})

    return results_df, preds_by_model


# -------------------------
# SEQ
# -------------------------

def prepare_seq_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    cfg: Config,
    run_id: str,
    force_recompute: bool,
) -> Dict:
    """Prepare sequential data loaders and scalers with caching.

    Returns dict:
        train_loader, val_loader, test_loader
        scaler_seq
        feature_cols_seq
        idx_seq_test
        n_total_test
        df_test_fe  (aligned with pointwise rows)
    """
    p_feat = artifact_path("fe", "feature_cols_seq", run_id, "json")
    p_scaler = artifact_path("models", "scaler_seq", run_id, "joblib")
    p_idx = artifact_path("splits", "idx_seq_test", run_id, "npz")
    p_meta = artifact_path("fe", "seq_meta", run_id, "json")

    # We do not cache dataloaders; we cache the stuff needed to rebuild deterministically.
    # If you want caching sequences, add outputs/fe/X_train_seqs_scaled__{run_id}.npz, etc.

    # Always ensure session_id
    df_train = add_session_id(df_train)
    df_test = add_session_id(df_test)

    _require_cols(df_train, ["session_id", "device", "motion", "segment_id", "label_X", "label_Y"], "df_train")
    _require_cols(df_test, ["session_id", "device", "motion", "segment_id", "label_X", "label_Y"], "df_test")

    # Fit wifi selector on train
    wifi_cols = fit_wifi_selector(
        df_train,
        prefixes=tuple(_get_cfg(cfg, "wifi_prefixes", ())),
        rssi_missing=float(_get_cfg(cfg, "rssi_missing", -100.0)),
        wifi_min_presence=float(_get_cfg(cfg, "wifi_min_presence", 0.05)),
        wifi_topk=int(_get_cfg(cfg, "wifi_topk", 10)),
    )

    # FE train/test
    df_train_fe, X_train_num, _, _, _ = feature_engineering_best(
        df_train,
        time_col=str(_get_cfg(cfg, "time_col", "t_ms")),
        imu_cols=tuple(_get_cfg(cfg, "imu_cols", ())),
        rolling_window_size=int(_get_cfg(cfg, "rolling_window_size", 10)),
        rolling_group_cols=tuple(_get_cfg(cfg, "rolling_group_cols", ("device", "motion"))),
        rolling_imu_cols=tuple(_get_cfg(cfg, "rolling_imu_cols", ())),
        rolling_stats=tuple(_get_cfg(cfg, "rolling_stats", ("mean", "var", "min", "max"))),
        add_rolling=bool(_get_cfg(cfg, "add_rolling", True)),
        add_diff=bool(_get_cfg(cfg, "add_diff", True)),
        add_dt_derivative=bool(_get_cfg(cfg, "add_dt_derivative", True)),
        eps_dt_ms=float(_get_cfg(cfg, "eps_dt_ms", 1.0)),
        wifi_cols_fixed=wifi_cols,
        wifi_prefixes=tuple(_get_cfg(cfg, "wifi_prefixes", ())),
        rssi_missing=float(_get_cfg(cfg, "rssi_missing", -100.0)),
        fill_numeric_with=str(_get_cfg(cfg, "fill_numeric_with", "median")),
        verbose=False,
    )

    df_test_fe, X_test_num, _, _, _ = feature_engineering_best(
        df_test,
        time_col=str(_get_cfg(cfg, "time_col", "t_ms")),
        imu_cols=tuple(_get_cfg(cfg, "imu_cols", ())),
        rolling_window_size=int(_get_cfg(cfg, "rolling_window_size", 10)),
        rolling_group_cols=tuple(_get_cfg(cfg, "rolling_group_cols", ("device", "motion"))),
        rolling_imu_cols=tuple(_get_cfg(cfg, "rolling_imu_cols", ())),
        rolling_stats=tuple(_get_cfg(cfg, "rolling_stats", ("mean", "var", "min", "max"))),
        add_rolling=bool(_get_cfg(cfg, "add_rolling", True)),
        add_diff=bool(_get_cfg(cfg, "add_diff", True)),
        add_dt_derivative=bool(_get_cfg(cfg, "add_dt_derivative", True)),
        eps_dt_ms=float(_get_cfg(cfg, "eps_dt_ms", 1.0)),
        wifi_cols_fixed=wifi_cols,
        wifi_prefixes=tuple(_get_cfg(cfg, "wifi_prefixes", ())),
        rssi_missing=float(_get_cfg(cfg, "rssi_missing", -100.0)),
        fill_numeric_with=str(_get_cfg(cfg, "fill_numeric_with", "median")),
        verbose=False,
    )

    # Optional: use only true anchor points as sequence targets to avoid staircase labels from asof-nearest.
    use_anchor_points = bool(_get_cfg(cfg, "seq_use_anchor_points", True))
    gap_thr_ms = _get_cfg(cfg, "gap_thr_ms", 1000.0)
    gap_thr_ms = float(gap_thr_ms) if gap_thr_ms is not None else None
    min_points = max(20, int(_get_cfg(cfg, "window_size", 20)) * 2)

    def _subset_anchor_points(df_fe: pd.DataFrame, X_num: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not use_anchor_points:
            return df_fe, X_num
        if not {"anchor_X", "anchor_Y"}.issubset(df_fe.columns):
            return df_fe, X_num
        m = np.isfinite(df_fe["anchor_X"].to_numpy(dtype=float)) & np.isfinite(df_fe["anchor_Y"].to_numpy(dtype=float))
        if int(m.sum()) < min_points:
            return df_fe, X_num
        d = df_fe.loc[m].copy()
        x = X_num.loc[m].copy()
        d["__rid"] = np.arange(len(d), dtype=int)
        d["label_X"] = d["anchor_X"].astype(float)
        d["label_Y"] = d["anchor_Y"].astype(float)
        if {"session_id", "t_ms"}.issubset(d.columns) and gap_thr_ms is not None:
            d = d.sort_values(["session_id", "t_ms"], kind="mergesort").reset_index(drop=True)
            x = x.iloc[d["__rid"].to_numpy(dtype=int)].reset_index(drop=True)
            dt = d.groupby("session_id")["t_ms"].diff()
            d["segment_id"] = (dt.isna() | (dt > gap_thr_ms)).groupby(d["session_id"]).cumsum().astype(int)
        d = d.drop(columns=["__rid"]).reset_index(drop=True)
        x = x.reset_index(drop=True)
        return d, x

    df_train_fe, X_train_num = _subset_anchor_points(df_train_fe, X_train_num)
    df_test_fe, X_test_num = _subset_anchor_points(df_test_fe, X_test_num)

    # Optional PCA on top correlated numeric features (train-only)
    use_pca = bool(_get_cfg(cfg, "seq_use_pca", False))
    pca_n = int(_get_cfg(cfg, "seq_pca_n_components", int(_get_cfg(cfg, "top_k", 20))))
    pca_topk = int(_get_cfg(cfg, "seq_pca_topk_corr", int(_get_cfg(cfg, "top_k", 20))))

    # Optional top-K correlation selection by group (train-only)
    use_topk_corr = bool(_get_cfg(cfg, "seq_use_topk_corr", False))
    topk_corr_k = int(_get_cfg(cfg, "seq_topk_corr_k", int(_get_cfg(cfg, "top_k", 20))))

    def _corr_score(series: pd.Series) -> float | None:
        s = pd.to_numeric(series, errors="coerce")
        if s.isna().all():
            return None
        try:
            corr_x = np.corrcoef(s.fillna(0.0), df_train_fe["label_X"].to_numpy())[0, 1]
            corr_y = np.corrcoef(s.fillna(0.0), df_train_fe["label_Y"].to_numpy())[0, 1]
            score = np.nanmean([abs(corr_x), abs(corr_y)])
            return float(score) if not np.isnan(score) else None
        except Exception:
            return None

    def _select_topk_by_group(cols: list[str], k: int) -> list[str]:
        if not cols:
            return []
        scores = {}
        for c in cols:
            score = _corr_score(X_train_num[c])
            if score is not None:
                scores[c] = score
        if not scores:
            return cols[:k]
        return sorted(scores, key=scores.get, reverse=True)[:k]

    if use_topk_corr:
        imu_cols = list(_get_cfg(cfg, "imu_cols", ()))
        roll_cols = [c for c in X_train_num.columns if "_roll_" in c]
        deriv_cols = [c for c in X_train_num.columns if c.endswith("_diff") or c.endswith("_dt")]
        wifi_cols = [c for c in X_train_num.columns if c not in set(imu_cols + roll_cols + deriv_cols)]

        k = max(1, topk_corr_k)
        sel = []
        sel += _select_topk_by_group([c for c in imu_cols if c in X_train_num.columns], k)
        sel += _select_topk_by_group(roll_cols, k)
        sel += _select_topk_by_group(deriv_cols, k)
        sel += _select_topk_by_group(wifi_cols, k)

        # keep order, drop duplicates
        seen = set()
        sel = [c for c in sel if (c in X_train_num.columns) and (c not in seen and not seen.add(c))]
        if sel:
            X_train_num = X_train_num[sel].copy()
            X_test_num = X_test_num.reindex(columns=sel, fill_value=0.0).copy()
            if run_id:
                save_json({"topk_corr_features": sel}, artifact_path("fe", "seq_topk_corr", run_id, "json"))

    if use_pca:
        # correlation-based feature selection (top-k by mean abs corr to label_X/Y)
        corr_scores = {}
        for col in X_train_num.columns:
            score = _corr_score(X_train_num[col])
            if score is not None:
                corr_scores[col] = score

        top_cols = sorted(corr_scores, key=corr_scores.get, reverse=True)[: max(1, pca_topk)]
        if not top_cols:
            top_cols = list(X_train_num.columns)

        scaler_pca = StandardScaler()
        Xtr_scaled = scaler_pca.fit_transform(X_train_num[top_cols])
        Xte_scaled = scaler_pca.transform(X_test_num[top_cols])

        n_comp = min(pca_n, Xtr_scaled.shape[1])
        pca = PCA(n_components=n_comp, random_state=int(_get_cfg(cfg, "random_seed", 42)))
        Xtr_pca = pca.fit_transform(Xtr_scaled)
        Xte_pca = pca.transform(Xte_scaled)

        X_train_num = pd.DataFrame(Xtr_pca, columns=[f"pca_{i}" for i in range(n_comp)])
        X_test_num = pd.DataFrame(Xte_pca, columns=[f"pca_{i}" for i in range(n_comp)])

        if run_id:
            save_json({"top_corr_features": top_cols}, artifact_path("fe", "seq_pca_topcorr", run_id, "json"))
            save_joblib(scaler_pca, artifact_path("models", "seq_pca_scaler", run_id, "joblib"))
            save_joblib(pca, artifact_path("models", "seq_pca", run_id, "joblib"))

    # Build tab for OHE
    time_col = str(_get_cfg(cfg, "time_col", "t_ms"))
    # Keep a single canonical time column in tabular seq frame (avoid duplicate `t_ms` labels).
    X_train_num_seq = X_train_num.drop(columns=[time_col], errors="ignore")
    X_test_num_seq = X_test_num.drop(columns=[time_col], errors="ignore")

    train_tab = pd.concat(
        [
            X_train_num_seq.reset_index(drop=True),
            df_train_fe[["device", "motion", "session_id", "segment_id", "label_X", "label_Y", time_col]].reset_index(drop=True),
        ],
        axis=1,
    )
    test_tab = pd.concat(
        [
            X_test_num_seq.reset_index(drop=True),
            df_test_fe[["device", "motion", "session_id", "segment_id", "label_X", "label_Y", time_col]].reset_index(drop=True),
        ],
        axis=1,
    )

    # Optional densification on a regular time grid to increase sequence count.
    densify_step_ms = _get_cfg(cfg, "seq_densify_step_ms", None)
    densify_step_ms = int(densify_step_ms) if densify_step_ms is not None else None
    densify_max_factor = float(_get_cfg(cfg, "seq_densify_max_factor", 4.0))

    def _densify_tab(tab: pd.DataFrame) -> pd.DataFrame:
        if densify_step_ms is None or densify_step_ms <= 0 or time_col not in tab.columns:
            return tab
        parts = []
        for _, g in tab.groupby(["session_id", "segment_id"], sort=False):
            g = g.sort_values(time_col, kind="mergesort")
            if len(g) < 2:
                parts.append(g)
                continue
            t = g[time_col].to_numpy(dtype=float)
            grid = np.arange(t[0], t[-1] + densify_step_ms, densify_step_ms, dtype=float)
            if len(grid) <= len(g) or len(grid) > int(np.ceil(len(g) * densify_max_factor)):
                parts.append(g)
                continue
            base = g.set_index(time_col)
            base = base[~base.index.duplicated(keep="last")]
            full = base.reindex(base.index.union(grid)).sort_index()
            num_cols = full.select_dtypes(include=[np.number]).columns.tolist()
            for c in ("segment_id",):
                if c in num_cols:
                    num_cols.remove(c)
            if num_cols:
                full[num_cols] = full[num_cols].interpolate(method="index", limit_direction="both")
            full["session_id"] = g["session_id"].iloc[0]
            full["segment_id"] = g["segment_id"].iloc[0]
            full["device"] = g["device"].iloc[0]
            full["motion"] = g["motion"].iloc[0]
            out = full.loc[grid].reset_index().rename(columns={"index": time_col})
            oh_cols = [c for c in out.columns if c.startswith("device_") or c.startswith("motion_")]
            if oh_cols:
                out[oh_cols] = (out[oh_cols] >= 0.5).astype(float)
            parts.append(out)
        return pd.concat(parts, ignore_index=True).sort_values(["session_id", "segment_id", time_col], kind="mergesort").reset_index(drop=True)

    train_tab = _densify_tab(train_tab)
    test_tab = _densify_tab(test_tab)
    df_test_ref = test_tab[["device", "motion", "session_id", "segment_id", "label_X", "label_Y", time_col]].copy()

    train_ohe = pd.get_dummies(train_tab, columns=["device", "motion"], drop_first=True)
    test_ohe = pd.get_dummies(test_tab, columns=["device", "motion"], drop_first=True)
    # features = all but labels + session_id
    feature_cols_seq = [c for c in train_ohe.columns if c not in {"label_X", "label_Y", "session_id", "segment_id"}]
    if not bool(_get_cfg(cfg, "seq_include_time_feature", False)) and time_col in feature_cols_seq:
        feature_cols_seq = [c for c in feature_cols_seq if c != time_col]

    # align test
    test_ohe = test_ohe.reindex(
        columns=feature_cols_seq + ["session_id", "segment_id", "label_X", "label_Y"],
        fill_value=0.0,
    )

    # sequences
    window_size = int(_get_cfg(cfg, "window_size", 20))
    min_window_size = int(_get_cfg(cfg, "seq_min_window_size", 4))
    min_test_windows = int(_get_cfg(cfg, "seq_min_test_windows", 80))
    min_coverage = float(_get_cfg(cfg, "seq_min_coverage", 0.25))
    target_mode = str(_get_cfg(cfg, "seq_target_mode", "abs"))

    # Ensure window_size is feasible given segment lengths
    seg_cols = ["session_id", "segment_id"]
    if all(c in df_train_fe.columns for c in seg_cols) and all(c in df_test_fe.columns for c in seg_cols):
        max_train = int(df_train_fe.groupby(seg_cols).size().max())
        max_test = int(df_test_fe.groupby(seg_cols).size().max())
        max_allowed = min(max_train, max_test)
        if max_allowed < window_size:
            if max_allowed < 2:
                raise ValueError(
                    "No sequences built: max segment length is < 2. "
                    "Increase gap_thr_ms/merge_tolerance_ms or disable strict drops."
                )
            window_size = max_allowed
            print(f"[WARN] window_size reduced to {window_size} (max segment length).")

    max_dt_ms_cfg = _get_cfg(cfg, "seq_max_dt_ms", None)
    max_dt_ms_cfg = float(max_dt_ms_cfg) if max_dt_ms_cfg is not None else None
    max_window_ms_cfg = _get_cfg(cfg, "seq_max_window_ms", None)
    max_window_ms_cfg = float(max_window_ms_cfg) if max_window_ms_cfg is not None else None

    ignore_time_checks = bool(_get_cfg(cfg, "seq_ignore_time_checks", True))

    def _build_for_window(w: int, dt_limit: float | None, win_limit: float | None):
        xtr, ytr, itr = build_sequences(
            train_ohe,
            feature_cols_seq,
            window_size=w,
            session_col="session_id",
            segment_col="segment_id",
            target_mode=target_mode,
            time_col=time_col,
            max_dt_ms=dt_limit,
            max_window_ms=win_limit,
            enforce_time_checks=not ignore_time_checks,
        )
        xte, yte, ite = build_sequences(
            test_ohe,
            feature_cols_seq,
            window_size=w,
            session_col="session_id",
            segment_col="segment_id",
            target_mode=target_mode,
            time_col=time_col,
            max_dt_ms=dt_limit,
            max_window_ms=win_limit,
            enforce_time_checks=not ignore_time_checks,
        )
        return xtr, ytr, itr, xte, yte, ite

    X_train_seqs, y_train_seqs, idx_train, X_test_seqs, y_test_seqs, idx_test = _build_for_window(
        window_size, max_dt_ms_cfg, max_window_ms_cfg
    )

    # If too strict, progressively relax constraints and shrink window.
    n_total_test = int(len(df_test_ref))
    while True:
        has_data = (X_train_seqs.size > 0) and (X_test_seqs.size > 0)
        coverage = (len(idx_test) / max(1, n_total_test))
        enough_windows = len(idx_test) >= min_test_windows
        enough_coverage = coverage >= min_coverage
        if has_data and enough_windows and enough_coverage:
            break

        relaxed = not (max_dt_ms_cfg is None and max_window_ms_cfg is None)
        if relaxed:
            max_dt_ms_cfg = None
            max_window_ms_cfg = None
            print("[WARN] Relaxing sequence dt/window constraints to increase coverage.")
        elif window_size > min_window_size:
            window_size = max(min_window_size, window_size // 2)
            print(f"[WARN] Reducing sequence window_size to {window_size} to increase test windows.")
        else:
            break

        X_train_seqs, y_train_seqs, idx_train, X_test_seqs, y_test_seqs, idx_test = _build_for_window(
            window_size, max_dt_ms_cfg, max_window_ms_cfg
        )

    if X_train_seqs.size == 0 or X_test_seqs.size == 0:
        max_train = int(df_train_fe.groupby(seg_cols).size().max())
        max_test = int(df_test_fe.groupby(seg_cols).size().max())
        raise ValueError(
            "No sequences built. "
            f"max seg_len train={max_train}, test={max_test}, window_size={window_size}. "
            "Reduce window_size or increase gap_thr_ms."
        )

    coverage = len(idx_test) / max(1, n_total_test)
    if len(idx_test) < min_test_windows or coverage < min_coverage:
        print(
            "[WARN] Low sequential coverage after fallback: "
            f"n_test_windows={len(idx_test)}, coverage={coverage:.3f}, "
            f"window_size={window_size}. Results may be unstable."
        )

    # scaler fit on train only
    scaler = fit_seq_scaler(X_train_seqs)
    X_train_scaled = transform_seq_scaler(X_train_seqs, scaler)
    X_test_scaled = transform_seq_scaler(X_test_seqs, scaler)

    # group split for val based on sequence target session_id
    seq_session_train = train_ohe.loc[idx_train, "session_id"].values
    g_tr, g_val = group_split_train_val(
        pd.DataFrame({"session_id": seq_session_train}),
        group_col="session_id",
        val_size=float(_get_cfg(cfg, "val_size", 0.2)),
        seed=int(_get_cfg(cfg, "random_seed", 42)),
    )
    tr_idx = g_tr.index.to_numpy()
    val_idx = g_val.index.to_numpy()
    if val_idx.size == 0:
        val_idx = tr_idx.copy()

    # loaders
    batch_size = int(_get_cfg(cfg, "batch_size", 64))

    train_loader = DataLoader(TrajDataset(X_train_scaled[tr_idx], y_train_seqs[tr_idx]),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TrajDataset(X_train_scaled[val_idx], y_train_seqs[val_idx]),
                            batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TrajDataset(X_test_scaled, y_test_seqs),
                             batch_size=batch_size, shuffle=False)

    # cache meta
    if force_recompute or (not all(map(exists, [p_feat, p_scaler, p_idx, p_meta]))):
        save_json({"feature_cols_seq": feature_cols_seq, "window_size": window_size}, p_feat)
        save_joblib(scaler, p_scaler)
        save_npz(p_idx, idx_seq_test=np.asarray(idx_test, dtype=int))
        save_json(
            {
                "n_total_test": int(len(df_test_ref)),
                "n_test_windows": int(len(idx_test)),
                "coverage": float(coverage),
                "window_size_effective": int(window_size),
                "ignore_time_checks": bool(ignore_time_checks),
                "densify_step_ms": densify_step_ms,
            },
            p_meta,
        )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "scaler_seq": scaler,
        "feature_cols_seq": feature_cols_seq,
        "idx_seq_test": idx_test,
        "n_total_test": int(len(df_test_ref)),
        "n_test_windows": int(len(idx_test)),
        "seq_coverage": float(coverage),
        "window_size_effective": int(window_size),
        "df_test_fe": df_test_ref.reset_index(drop=True),
        "seq_target_mode": target_mode,
        "run_id": run_id,
        "force_recompute": force_recompute,
    }


def eval_seq_models(seq_bundle: Dict, *, cfg: Config) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Train/evaluate sequence models and return metrics and pointwise preds.

    Same note as tabular: signature lacks run_id/force_recompute. We fallback to cfg.run_id/cfg.force_recompute if present.
    """
    run_id = seq_bundle.get("run_id", getattr(cfg, "run_id", None))
    force_recompute = bool(seq_bundle.get("force_recompute", getattr(cfg, "force_recompute", False)))

    if run_id is not None:
        p_results_pkl = artifact_path("metrics", "results_seq", run_id, "joblib")
        p_results_csv = artifact_path("metrics", "results_seq", run_id, "csv")
        p_preds = artifact_path("metrics", "preds_seq_pointwise", run_id, "npz")
    else:
        p_results_pkl = p_results_csv = p_preds = None

    thresholds = tuple(_get_cfg(cfg, "thresholds", (0.25, 0.5, 1.0, 2.0)))
    use_kalman = bool(_get_cfg(cfg, "use_kalman_postproc", True))
    use_guardrails = bool(_get_cfg(cfg, "use_pred_guardrails", True))
    time_col = str(_get_cfg(cfg, "time_col", "t_ms"))
    max_speed_mps = _get_cfg(cfg, "baseline_max_speed_mps", 2.5)
    max_speed_mps = float(max_speed_mps) if max_speed_mps is not None else None
    max_dt_ms = _get_cfg(cfg, "gap_thr_ms", 1000.0)
    max_dt_ms = float(max_dt_ms) if max_dt_ms is not None else None
    kalman_process_var = float(_get_cfg(cfg, "kalman_process_var", 1e-3))
    kalman_meas_var = float(_get_cfg(cfg, "kalman_meas_var", 1e-1))

    train_loader = seq_bundle["train_loader"]
    val_loader = seq_bundle["val_loader"]
    test_loader = seq_bundle["test_loader"]
    idx_seq_test = seq_bundle["idx_seq_test"]
    n_total_test = seq_bundle["n_total_test"]
    df_test_base = seq_bundle.get("df_test_fe")
    seq_target_mode = str(seq_bundle.get("seq_target_mode", "abs"))

    def _seq_pack(y_true_arr: np.ndarray, y_pred_arr: np.ndarray) -> Dict[str, np.ndarray]:
        pack: Dict[str, np.ndarray] = {"y_true": y_true_arr, "y_pred": y_pred_arr}
        if df_test_base is not None:
            ref = df_test_base.loc[idx_seq_test].reset_index(drop=True)
            if "t_ms" in ref.columns:
                pack["t_ms"] = ref["t_ms"].to_numpy(dtype=float)
            if "session_id" in ref.columns:
                pack["session_id"] = ref["session_id"].to_numpy()
            if "segment_id" in ref.columns:
                pack["segment_id"] = ref["segment_id"].to_numpy()
            if "anchor_X" in ref.columns and "anchor_Y" in ref.columns:
                pack["anchor_X"] = ref["anchor_X"].to_numpy(dtype=float)
                pack["anchor_Y"] = ref["anchor_Y"].to_numpy(dtype=float)
        return pack

    # infer input dim D
    xb0 = next(iter(train_loader))[0]
    D = int(xb0.shape[-1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    preds_pointwise: Dict[str, np.ndarray] = {}
    preds_seq: Dict[str, Dict[str, np.ndarray]] = {}
    target_weights = None
    if seq_target_mode == "delta":
        y_train_delta = train_loader.dataset.y.detach().cpu().numpy()
        std_xy = np.std(y_train_delta, axis=0)
        std_xy = np.maximum(std_xy, 1e-6)
        inv = 1.0 / std_xy
        target_weights = inv / np.mean(inv)

    # cached metrics + pointwise preds, but still build seq preds if checkpoints exist
    if run_id is not None and (not force_recompute) and all(map(exists, [p_results_pkl, p_preds])):
        metrics_df = load_joblib(p_results_pkl)
        preds_pack = load_npz(p_preds)
        preds_pointwise = _dict_npz_to_preds(preds_pack)
        bad_shape = any(
            (np.asarray(v).ndim != 2) or (np.asarray(v).shape[0] != n_total_test) or (np.asarray(v).shape[1] != 2)
            for v in preds_pointwise.values()
        )
        if bad_shape:
            force_recompute = True
        else:
            ckpt_lstm = artifact_path("models", "best_lstm", run_id, "pth")
            ckpt_gru = artifact_path("models", "best_gru", run_id, "pth")

            cache_ok = True

            if exists(ckpt_lstm):
                lstm_params = dict(_get_cfg(cfg, "lstm_params", {}))
                lstm = LSTMRegressorDiamond(input_dim=D, **lstm_params).to(device)
                try:
                    lstm.load_state_dict(torch.load(ckpt_lstm, map_location=device, weights_only=True))
                    y_true, y_pred = predict_torch_model(lstm, test_loader, device)
                    if seq_target_mode == "delta" and df_test_base is not None:
                        y_true_pos = df_test_base.loc[idx_seq_test, ["label_X", "label_Y"]].to_numpy(dtype=float)
                        y_pred_pos = _reconstruct_positions_from_deltas(df_test_base, idx_seq_test, y_pred)
                        if use_guardrails:
                            y_pred_pos = _stabilize_predictions_by_group(
                                y_pred_pos,
                                df_test_base.loc[idx_seq_test].reset_index(drop=True),
                                time_col=time_col,
                                max_speed_mps=max_speed_mps,
                                max_dt_ms=max_dt_ms,
                            )
                        preds_seq["LSTM_FE"] = _seq_pack(y_true_pos, y_pred_pos)
                    else:
                        if use_guardrails and df_test_base is not None:
                            y_pred = _stabilize_predictions_by_group(
                                y_pred,
                                df_test_base.loc[idx_seq_test].reset_index(drop=True),
                                time_col=time_col,
                                max_speed_mps=max_speed_mps,
                                max_dt_ms=max_dt_ms,
                            )
                        preds_seq["LSTM_FE"] = _seq_pack(y_true, y_pred)
                except RuntimeError:
                    cache_ok = False

            if exists(ckpt_gru):
                gru_params = dict(_get_cfg(cfg, "gru_params", {}))
                gru = GRURegressorDiamond(input_dim=D, **gru_params).to(device)
                try:
                    gru.load_state_dict(torch.load(ckpt_gru, map_location=device, weights_only=True))
                    y_true, y_pred = predict_torch_model(gru, test_loader, device)
                    if seq_target_mode == "delta" and df_test_base is not None:
                        y_true_pos = df_test_base.loc[idx_seq_test, ["label_X", "label_Y"]].to_numpy(dtype=float)
                        y_pred_pos = _reconstruct_positions_from_deltas(df_test_base, idx_seq_test, y_pred)
                        if use_guardrails:
                            y_pred_pos = _stabilize_predictions_by_group(
                                y_pred_pos,
                                df_test_base.loc[idx_seq_test].reset_index(drop=True),
                                time_col=time_col,
                                max_speed_mps=max_speed_mps,
                                max_dt_ms=max_dt_ms,
                            )
                        preds_seq["GRU_FE"] = _seq_pack(y_true_pos, y_pred_pos)
                    else:
                        if use_guardrails and df_test_base is not None:
                            y_pred = _stabilize_predictions_by_group(
                                y_pred,
                                df_test_base.loc[idx_seq_test].reset_index(drop=True),
                                time_col=time_col,
                                max_speed_mps=max_speed_mps,
                                max_dt_ms=max_dt_ms,
                            )
                        preds_seq["GRU_FE"] = _seq_pack(y_true, y_pred)
                except RuntimeError:
                    cache_ok = False

            if cache_ok:
                return metrics_df, preds_pointwise, preds_seq
        # fallback: force recompute if checkpoint dims mismatch
        force_recompute = True

    # LSTM
    lstm_params = dict(_get_cfg(cfg, "lstm_params", {}))
    lstm = LSTMRegressorDiamond(input_dim=D, **lstm_params).to(device)

    ckpt_lstm = artifact_path("models", "best_lstm", run_id, "pth") if run_id is not None else None
    train_torch_model(
        lstm,
        train_loader,
        val_loader,
        epochs=int(_get_cfg(cfg, "epochs", 100)),
        patience=int(_get_cfg(cfg, "patience", 20)),
        lr=float(_get_cfg(cfg, "lr", 5e-4)),
        device=device,
        ckpt_path=ckpt_lstm,
        target_weights=target_weights,
    )
    y_true, y_pred = predict_torch_model(lstm, test_loader, device)
    if seq_target_mode == "delta" and df_test_base is not None:
        y_true_pos = df_test_base.loc[idx_seq_test, ["label_X", "label_Y"]].to_numpy(dtype=float)
        y_pred_pos = _reconstruct_positions_from_deltas(df_test_base, idx_seq_test, y_pred)
        if use_guardrails:
            y_pred_pos = _stabilize_predictions_by_group(
                y_pred_pos,
                df_test_base.loc[idx_seq_test].reset_index(drop=True),
                time_col=time_col,
                max_speed_mps=max_speed_mps,
                max_dt_ms=max_dt_ms,
            )
        rows.append(evaluate_regression(y_true_pos, y_pred_pos, "LSTM_FE", thresholds=thresholds))
        preds_pointwise["LSTM_FE"] = seq_preds_to_pointwise(y_pred_pos, idx_seq_test, n_total_test, agg="mean")
        preds_seq["LSTM_FE"] = _seq_pack(y_true_pos, y_pred_pos)
        if use_kalman:
            y_pred_pos_kf = _apply_kalman_by_group(
                y_pred_pos,
                df_test_base.loc[idx_seq_test].reset_index(drop=True),
                time_col=time_col,
                process_var=kalman_process_var,
                meas_var=kalman_meas_var,
            )
            rows.append(evaluate_regression(y_true_pos, y_pred_pos_kf, "LSTM_FE_KF", thresholds=thresholds))
            preds_pointwise["LSTM_FE_KF"] = seq_preds_to_pointwise(y_pred_pos_kf, idx_seq_test, n_total_test, agg="mean")
            preds_seq["LSTM_FE_KF"] = _seq_pack(y_true_pos, y_pred_pos_kf)
    else:
        if use_guardrails and df_test_base is not None:
            y_pred = _stabilize_predictions_by_group(
                y_pred,
                df_test_base.loc[idx_seq_test].reset_index(drop=True),
                time_col=time_col,
                max_speed_mps=max_speed_mps,
                max_dt_ms=max_dt_ms,
            )
        rows.append(evaluate_regression(y_true, y_pred, "LSTM_FE", thresholds=thresholds))
        preds_pointwise["LSTM_FE"] = seq_preds_to_pointwise(y_pred, idx_seq_test, n_total_test, agg="mean")
        preds_seq["LSTM_FE"] = _seq_pack(y_true, y_pred)

    # GRU
    gru_params = dict(_get_cfg(cfg, "gru_params", {}))
    gru = GRURegressorDiamond(input_dim=D, **gru_params).to(device)

    ckpt_gru = artifact_path("models", "best_gru", run_id, "pth") if run_id is not None else None
    train_torch_model(
        gru,
        train_loader,
        val_loader,
        epochs=int(_get_cfg(cfg, "epochs", 100)),
        patience=int(_get_cfg(cfg, "gru_patience", int(_get_cfg(cfg, "patience", 20)))),
        lr=float(_get_cfg(cfg, "gru_lr", float(_get_cfg(cfg, "lr", 5e-4)))),
        device=device,
        ckpt_path=ckpt_gru,
        target_weights=target_weights,
    )
    y_true, y_pred = predict_torch_model(gru, test_loader, device)
    if seq_target_mode == "delta" and df_test_base is not None:
        y_true_pos = df_test_base.loc[idx_seq_test, ["label_X", "label_Y"]].to_numpy(dtype=float)
        y_pred_pos = _reconstruct_positions_from_deltas(df_test_base, idx_seq_test, y_pred)
        if use_guardrails:
            y_pred_pos = _stabilize_predictions_by_group(
                y_pred_pos,
                df_test_base.loc[idx_seq_test].reset_index(drop=True),
                time_col=time_col,
                max_speed_mps=max_speed_mps,
                max_dt_ms=max_dt_ms,
            )
        rows.append(evaluate_regression(y_true_pos, y_pred_pos, "GRU_FE", thresholds=thresholds))
        preds_pointwise["GRU_FE"] = seq_preds_to_pointwise(y_pred_pos, idx_seq_test, n_total_test, agg="mean")
        preds_seq["GRU_FE"] = _seq_pack(y_true_pos, y_pred_pos)
        if use_kalman:
            y_pred_pos_kf = _apply_kalman_by_group(
                y_pred_pos,
                df_test_base.loc[idx_seq_test].reset_index(drop=True),
                time_col=time_col,
                process_var=kalman_process_var,
                meas_var=kalman_meas_var,
            )
            rows.append(evaluate_regression(y_true_pos, y_pred_pos_kf, "GRU_FE_KF", thresholds=thresholds))
            preds_pointwise["GRU_FE_KF"] = seq_preds_to_pointwise(y_pred_pos_kf, idx_seq_test, n_total_test, agg="mean")
            preds_seq["GRU_FE_KF"] = _seq_pack(y_true_pos, y_pred_pos_kf)
    else:
        if use_guardrails and df_test_base is not None:
            y_pred = _stabilize_predictions_by_group(
                y_pred,
                df_test_base.loc[idx_seq_test].reset_index(drop=True),
                time_col=time_col,
                max_speed_mps=max_speed_mps,
                max_dt_ms=max_dt_ms,
            )
        rows.append(evaluate_regression(y_true, y_pred, "GRU_FE", thresholds=thresholds))
        preds_pointwise["GRU_FE"] = seq_preds_to_pointwise(y_pred, idx_seq_test, n_total_test, agg="mean")
        preds_seq["GRU_FE"] = _seq_pack(y_true, y_pred)

    metrics_df = pd.DataFrame(rows).sort_values("median_err_m").reset_index(drop=True)

    if run_id is not None:
        metrics_df.to_csv(p_results_csv, index=False)
        save_joblib(metrics_df, p_results_pkl)
        save_npz(p_preds, **{k: v for k, v in preds_pointwise.items()})

    return metrics_df, preds_pointwise, preds_seq


# -------------------------
# CROSS-DEVICE + SESSION DIAGNOSTICS
# -------------------------

def build_and_eval_tabular_cross_device(
    dfA: pd.DataFrame, dfB: pd.DataFrame, *, cfg: Config, model_name: str
) -> Dict:
    """Train on dfA, test on dfB for cross-device evaluation.

    This is a minimal faithful version of your monolith logic:
    - FE on each
    - remove 'device' from features (to mimic unknown device)
    - keep motion as categorical
    - train RF or XGB
    Returns dict with name, devices, metrics, y_true, y_pred, t_ms if available.
    """
    # FE
    wifi_cols = fit_wifi_selector(
        dfA,
        prefixes=tuple(_get_cfg(cfg, "wifi_prefixes", ())),
        rssi_missing=float(_get_cfg(cfg, "rssi_missing", -100.0)),
        wifi_min_presence=float(_get_cfg(cfg, "wifi_min_presence", 0.05)),
        wifi_topk=int(_get_cfg(cfg, "wifi_topk", 10)),
    )

    dfA_fe, XA_num, _, _, _ = feature_engineering_best(
        dfA,
        time_col=str(_get_cfg(cfg, "time_col", "t_ms")),
        imu_cols=tuple(_get_cfg(cfg, "imu_cols", ())),
        rolling_window_size=int(_get_cfg(cfg, "rolling_window_size", 10)),
        rolling_group_cols=tuple(_get_cfg(cfg, "rolling_group_cols", ("device", "motion"))),
        rolling_imu_cols=tuple(_get_cfg(cfg, "rolling_imu_cols", ())),
        rolling_stats=tuple(_get_cfg(cfg, "rolling_stats", ("mean", "var", "min", "max"))),
        add_rolling=bool(_get_cfg(cfg, "add_rolling", True)),
        add_diff=bool(_get_cfg(cfg, "add_diff", True)),
        add_dt_derivative=bool(_get_cfg(cfg, "add_dt_derivative", True)),
        eps_dt_ms=float(_get_cfg(cfg, "eps_dt_ms", 1.0)),
        wifi_cols_fixed=wifi_cols,
        wifi_prefixes=tuple(_get_cfg(cfg, "wifi_prefixes", ())),
        rssi_missing=float(_get_cfg(cfg, "rssi_missing", -100.0)),
        fill_numeric_with=str(_get_cfg(cfg, "fill_numeric_with", "median")),
        verbose=False,
    )
    dfB_fe, XB_num, _, _, _ = feature_engineering_best(
        dfB,
        time_col=str(_get_cfg(cfg, "time_col", "t_ms")),
        imu_cols=tuple(_get_cfg(cfg, "imu_cols", ())),
        rolling_window_size=int(_get_cfg(cfg, "rolling_window_size", 10)),
        rolling_group_cols=tuple(_get_cfg(cfg, "rolling_group_cols", ("device", "motion"))),
        rolling_imu_cols=tuple(_get_cfg(cfg, "rolling_imu_cols", ())),
        rolling_stats=tuple(_get_cfg(cfg, "rolling_stats", ("mean", "var", "min", "max"))),
        add_rolling=bool(_get_cfg(cfg, "add_rolling", True)),
        add_diff=bool(_get_cfg(cfg, "add_diff", True)),
        add_dt_derivative=bool(_get_cfg(cfg, "add_dt_derivative", True)),
        eps_dt_ms=float(_get_cfg(cfg, "eps_dt_ms", 1.0)),
        wifi_cols_fixed=wifi_cols,
        wifi_prefixes=tuple(_get_cfg(cfg, "wifi_prefixes", ())),
        rssi_missing=float(_get_cfg(cfg, "rssi_missing", -100.0)),
        fill_numeric_with=str(_get_cfg(cfg, "fill_numeric_with", "median")),
        verbose=False,
    )

    yA = dfA_fe[["label_X", "label_Y"]].to_numpy(dtype=float)
    yB = dfB_fe[["label_X", "label_Y"]].to_numpy(dtype=float)

    # Remove device to simulate unknown device at test; keep motion cat
    XA = pd.concat([XA_num.reset_index(drop=True), dfA_fe[["motion"]].reset_index(drop=True)], axis=1)
    XB = pd.concat([XB_num.reset_index(drop=True), dfB_fe[["motion"]].reset_index(drop=True)], axis=1)
    XB = XB.reindex(columns=XA.columns, fill_value=0.0)

    cat_cols = ["motion"]
    num_cols = [c for c in XA.columns if c not in cat_cols]
    preproc = build_preprocessor(num_cols, cat_cols)

    train_device = str(dfA_fe["device"].iloc[0]) if "device" in dfA_fe.columns else "A"
    test_device = str(dfB_fe["device"].iloc[0]) if "device" in dfB_fe.columns else "B"
    name = f"{model_name}_{train_device}_to_{test_device}"

    if model_name.upper() == "RF":
        model = train_rf(XA, yA, preproc, rf_params=dict(_get_cfg(cfg, "rf_params", {})))
        y_pred = model.predict(XB)
    elif model_name.upper() == "XGB":
        xgb_x, xgb_y = train_xgb_xy(XA, yA, preproc, xgb_params=dict(_get_cfg(cfg, "xgb_params", {})))
        y_pred = predict_xy_from_xgb(xgb_x, xgb_y, XB)
        model = (xgb_x, xgb_y)
    else:
        raise ValueError("model_name must be 'RF' or 'XGB'")

    if bool(_get_cfg(cfg, "use_pred_guardrails", True)):
        y_pred = _stabilize_predictions_by_group(
            y_pred,
            dfB_fe,
            time_col=str(_get_cfg(cfg, "time_col", "t_ms")),
            max_speed_mps=float(_get_cfg(cfg, "baseline_max_speed_mps", 2.5)),
            max_dt_ms=float(_get_cfg(cfg, "gap_thr_ms", 1000.0)),
        )

    metrics = evaluate_regression(yB, y_pred, name=name, thresholds=tuple(_get_cfg(cfg, "thresholds", (0.25, 0.5, 1.0, 2.0))))

    t_ms = dfB_fe["t_ms"].to_numpy() if "t_ms" in dfB_fe.columns else None

    return {
        "name": name,
        "train_device": train_device,
        "test_device": test_device,
        "metrics": metrics,
        "y_true": yB,
        "y_pred": np.asarray(y_pred, dtype=float),
        "t_ms": t_ms,
        "session_id": dfB_fe["session_id"].to_numpy() if "session_id" in dfB_fe.columns else None,
        "segment_id": dfB_fe["segment_id"].to_numpy() if "segment_id" in dfB_fe.columns else None,
    }


def session_analysis_device_motion(
    df_test_base: pd.DataFrame,
    preds_by_model: Dict,
    *,
    cfg: Config,
    min_points_per_session: int = 50,
    top_k_worst_sessions: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-session diagnostics and p95 pivot table.

    df_test_base: must contain device, motion, label_X, label_Y. Row order MUST match preds.
    preds_by_model: dict model_name -> y_pred (N,2) aligned with df_test_base rows.
    """
    _require_cols(df_test_base, ["device", "motion", "label_X", "label_Y"], "df_test_base")
    df = df_test_base.copy().reset_index(drop=True)
    if "session_id" not in df.columns:
        df["session_id"] = df["device"].astype(str) + "__" + df["motion"].astype(str)

    y_true_all = df[["label_X", "label_Y"]].to_numpy(dtype=float)
    N = len(df)

    # check preds alignment
    for m, y_pred in preds_by_model.items():
        y_pred = np.asarray(y_pred)
        if y_pred.shape != (N, 2):
            raise ValueError(f"{m}: y_pred must be (N,2) aligned with df_test_base (N={N}), got {y_pred.shape}")

    rows = []
    thresholds = tuple(_get_cfg(cfg, "thresholds", (0.25, 0.5, 1.0, 2.0)))

    for sid, idx in df.groupby("session_id").groups.items():
        idx = np.asarray(list(idx), dtype=int)
        if idx.size < min_points_per_session:
            continue

        dev = str(df.loc[idx[0], "device"])
        mot = str(df.loc[idx[0], "motion"])
        y_true_s = y_true_all[idx]

        for model_name, y_pred_all in preds_by_model.items():
            y_pred_s = np.asarray(y_pred_all)[idx]
            res = evaluate_regression(y_true_s, y_pred_s, name=model_name, thresholds=thresholds)
            res.update({"session_id": sid, "device": dev, "motion": mot})
            rows.append(res)

    session_df_long = pd.DataFrame(rows)

    p95_pivot = (
        session_df_long
        .pivot_table(index=["session_id", "device", "motion"], columns="model", values="p95_err_m", aggfunc="first")
        .reset_index()
    )

    # Optional: cache if cfg.run_id exists
    run_id = getattr(cfg, "run_id", None)
    force_recompute = bool(getattr(cfg, "force_recompute", False))
    if run_id is not None:
        p_long = artifact_path("metrics", "session_long", run_id, "joblib")
        p_pivot = artifact_path("metrics", "p95_pivot", run_id, "csv")
        if force_recompute or (not exists(p_long)):
            save_joblib(session_df_long, p_long)
            p95_pivot.to_csv(p_pivot, index=False)

    return session_df_long, p95_pivot
