"""Feature engineering utilities."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def get_wifi_columns(df: pd.DataFrame, prefixes: Tuple[str, ...]) -> List[str]:
    return [c for c in df.columns if any(str(c).lower().startswith(p) for p in prefixes)]


def fit_wifi_selector(
    df: pd.DataFrame,
    prefixes: Tuple[str, ...],
    *,
    rssi_missing: float,
    wifi_min_presence: float,
    wifi_topk: int,
) -> List[str]:
    """Fit Wi-Fi selector on training data and return fixed column list."""

    wifi_all_cols = get_wifi_columns(df, prefixes)
    if not wifi_all_cols:
        return []

    wifi_raw = df[wifi_all_cols].copy()
    presence_rate = wifi_raw.notna().mean()
    wifi_filled = wifi_raw.fillna(rssi_missing)
    wifi_var = wifi_filled.var()

    score = wifi_var * presence_rate
    score = score[presence_rate >= wifi_min_presence].sort_values(ascending=False)
    return score.head(wifi_topk).index.tolist()


def feature_engineering_best(
    df_in: pd.DataFrame,
    *,
    time_col: str,
    imu_cols: Tuple[str, ...],
    rolling_window_size: int,
    rolling_group_cols: Tuple[str, ...],
    rolling_imu_cols: Tuple[str, ...],
    rolling_stats: Tuple[str, ...],
    add_rolling: bool,
    add_diff: bool,
    add_dt_derivative: bool,
    eps_dt_ms: float,
    wifi_cols_fixed: List[str] | None,
    wifi_prefixes: Tuple[str, ...],
    rssi_missing: float,
    fill_numeric_with: str,
    verbose: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, Dict, List[str]]:
    """Return enriched df, features, labels, meta, and ordered feature columns."""

    df = df_in.copy()

    if time_col not in df.columns:
        raise ValueError(f"Missing time column: {time_col}")

    imu_cols_present = [c for c in imu_cols if c in df.columns]
    if len(imu_cols_present) == 0:
        raise ValueError("No IMU columns found in dataframe")

    def _fill_series(series: pd.Series) -> pd.Series:
        if not series.isna().any():
            return series
        if fill_numeric_with == "median":
            med = series.median()
            if pd.isna(med):
                med = 0.0
            return series.fillna(med)
        if fill_numeric_with == "zero":
            return series.fillna(0.0)
        raise ValueError("fill_numeric_with must be 'median' or 'zero'")

    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[time_col] = _fill_series(df[time_col])

    for c in imu_cols_present:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = _fill_series(df[c])

    if all(gc in df.columns for gc in rolling_group_cols):
        df = df.sort_values(list(rolling_group_cols) + [time_col], kind="mergesort").reset_index(drop=True)
    else:
        df = df.sort_values([time_col], kind="mergesort").reset_index(drop=True)

    roll_cols: List[str] = []
    if add_rolling:
        rolling_imu_cols_present = [c for c in rolling_imu_cols if c in df.columns]
        if rolling_imu_cols_present:
            if verbose:
                LOGGER.info(
                    "[FE] Rolling window=%s on %s IMU cols",
                    rolling_window_size,
                    len(rolling_imu_cols_present),
                )
            grp = (
                df.groupby(list(rolling_group_cols), sort=False)
                if all(gc in df.columns for gc in rolling_group_cols)
                else None
            )

            def _roll_transform(series: pd.Series, fn: str) -> pd.Series:
                r = series.rolling(window=rolling_window_size, min_periods=1)
                if fn == "mean":
                    return r.mean()
                if fn == "var":
                    return r.var()
                if fn == "min":
                    return r.min()
                if fn == "max":
                    return r.max()
                raise ValueError(f"Unknown rolling stat: {fn}")

            for col in rolling_imu_cols_present:
                for stat in rolling_stats:
                    newc = f"{col}_roll_{stat}"
                    if grp is not None:
                        df[newc] = grp[col].transform(lambda x, st=stat: _roll_transform(x, st))
                    else:
                        df[newc] = _roll_transform(df[col], stat)
                    roll_cols.append(newc)

            for c in roll_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                if "roll_var" in c:
                    df[c] = df[c].fillna(0.0)
                else:
                    med = df[c].median()
                    if pd.isna(med):
                        med = 0.0
                    df[c] = df[c].fillna(med)

    dt_ms = df[time_col].diff()
    dt_med = dt_ms.median()
    dt_ms = dt_ms.fillna(dt_med if not pd.isna(dt_med) else 10.0)
    dt_ms = dt_ms.clip(lower=eps_dt_ms)
    dt_s = dt_ms / 1000.0

    deriv_cols: List[str] = []
    if add_diff:
        for c in imu_cols_present:
            newc = f"{c}_diff"
            df[newc] = df[c].diff().fillna(0.0)
            deriv_cols.append(newc)

    if add_dt_derivative:
        for c in imu_cols_present:
            newc = f"{c}_dt"
            df[newc] = (df[c].diff() / dt_s).replace([np.inf, -np.inf], np.nan)
            df[newc] = _fill_series(df[newc]).fillna(0.0)
            deriv_cols.append(newc)

    wifi_all_cols = get_wifi_columns(df, wifi_prefixes)
    if wifi_cols_fixed is None:
        topk_wifi_cols = list(wifi_all_cols)
    else:
        topk_wifi_cols = list(wifi_cols_fixed)

    if topk_wifi_cols:
        for c in topk_wifi_cols:
            if c not in df.columns:
                df[c] = rssi_missing
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(rssi_missing).astype(float)

    feature_cols: List[str] = [time_col] + imu_cols_present
    feature_cols += deriv_cols
    feature_cols += roll_cols
    feature_cols += topk_wifi_cols

    seen = set()
    feature_cols = [c for c in feature_cols if (c in df.columns) and (c not in seen and not seen.add(c))]
    X_fe = df[feature_cols].copy()

    for c in topk_wifi_cols:
        if c in X_fe.columns and X_fe[c].isna().any():
            X_fe[c] = X_fe[c].fillna(rssi_missing)

    for c in X_fe.columns:
        if X_fe[c].dtype.kind in "biufc" and X_fe[c].isna().any():
            X_fe[c] = _fill_series(X_fe[c])

    nan_total = int(X_fe.isna().sum().sum())

    y_df = None
    if {"label_X", "label_Y"}.issubset(df.columns):
        y_df = df[["label_X", "label_Y"]].copy()

    meta = {
        "imu_cols": imu_cols_present,
        "derivative_cols": deriv_cols,
        "roll_cols": roll_cols,
        "wifi_all_cols_count": len(wifi_all_cols),
        "wifi_topk_cols": topk_wifi_cols,
        "feature_cols_final": feature_cols,
        "nan_total_in_X_fe": nan_total,
        "rolling_window_size": rolling_window_size,
        "rolling_group_cols": rolling_group_cols,
    }

    if verbose:
        LOGGER.info(
            "[FE] IMU=%s deriv=%s roll=%s wifi_all=%s wifi_topk=%s X_fe=%s nan=%s",
            len(imu_cols_present),
            len(deriv_cols),
            len(roll_cols),
            len(wifi_all_cols),
            len(topk_wifi_cols),
            X_fe.shape,
            nan_total,
        )

    return df, X_fe, y_df, meta, feature_cols

