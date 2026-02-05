"""Feature engineering utilities."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def get_wifi_columns(df: pd.DataFrame, prefixes: Tuple[str, ...]) -> List[str]:
    return [c for c in df.columns if any(str(c).lower().startswith(p) for p in prefixes)]


def fit_feature_normalizer(
    X: pd.DataFrame,
    *,
    device_series: pd.Series | None = None,
    min_std: float = 1e-6,
) -> Dict[str, object] | None:
    """Fit a feature-wise normalizer (global and optionally per-device)."""

    if X is None or len(X) == 0:
        return None

    X_num = X.apply(pd.to_numeric, errors="coerce")
    global_mean = X_num.mean()
    global_std = X_num.std(ddof=0).replace(0.0, np.nan).fillna(float(min_std))

    table: Dict[str, object] = {
        "min_std": float(min_std),
        "global": {
            "mean": global_mean.to_dict(),
            "std": global_std.to_dict(),
        },
        "per_device": {},
    }

    if device_series is None:
        return table

    device_series = pd.Series(device_series, index=X.index)
    for dev in device_series.dropna().unique():
        mask = device_series == dev
        X_dev = X_num.loc[mask]
        if X_dev.empty:
            continue
        mean = X_dev.mean()
        std = X_dev.std(ddof=0).replace(0.0, np.nan).fillna(float(min_std))
        table["per_device"][str(dev)] = {
            "mean": mean.to_dict(),
            "std": std.to_dict(),
        }

    return table


def apply_feature_normalizer(
    X: pd.DataFrame,
    table: Dict[str, object] | None,
    *,
    device_series: pd.Series | None = None,
) -> pd.DataFrame:
    """Apply a fitted feature normalizer to a dataframe."""

    if table is None or X is None or len(X) == 0:
        return X

    out = X.copy()
    cols = out.columns.tolist()

    global_stats = table.get("global", {}) if isinstance(table, dict) else {}
    global_mean = pd.Series(global_stats.get("mean", {}), dtype=float).reindex(cols).fillna(0.0)
    global_std = pd.Series(global_stats.get("std", {}), dtype=float).reindex(cols).fillna(1.0)
    min_std = float(table.get("min_std", 1e-6)) if isinstance(table, dict) else 1e-6
    global_std = global_std.replace(0.0, np.nan).fillna(min_std)

    base = out.copy()
    out = (base - global_mean) / global_std

    if device_series is None:
        return out

    device_series = pd.Series(device_series, index=out.index)
    per_device = table.get("per_device", {}) if isinstance(table, dict) else {}

    for dev in device_series.dropna().unique():
        mask = device_series == dev
        stats = per_device.get(str(dev), {})
        if not stats:
            continue
        mean = pd.Series(stats.get("mean", {}), dtype=float).reindex(cols).fillna(global_mean)
        std = pd.Series(stats.get("std", {}), dtype=float).reindex(cols).fillna(global_std)
        std = std.replace(0.0, np.nan).fillna(min_std)
        out.loc[mask, cols] = (base.loc[mask, cols] - mean) / std

    return out


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


def build_device_calibration_table(
    df: pd.DataFrame,
    *,
    ref_device: str,
    imu_cols: Tuple[str, ...],
    wifi_prefixes: Tuple[str, ...],
    xy_tol: float = 0.2,
    y_range: Tuple[float, float] = (0.0, 1.0),
    min_rows: int = 20,
    quantile_clip: Tuple[float, float] | None = (0.02, 0.98),
    shrink_k: int = 100,
    scale_clip: Tuple[float, float] | None = (0.2, 5.0),
    min_scale: float = 1e-6,
) -> Dict[str, object] | None:
    """Build per-device calibration coefficients from label anchors.

    WiFi: offsets at (0,0) with robust medians
    IMU: affine alignment along segment (0,0)->(0,1) using robust scale (IQR/std)
    """

    if df is None or len(df) == 0:
        return None
    if "device" not in df.columns or "label_X" not in df.columns or "label_Y" not in df.columns:
        return None

    devices = sorted(df["device"].dropna().unique().tolist())
    if ref_device not in devices:
        return None

    wifi_cols = get_wifi_columns(df, wifi_prefixes)
    imu_cols_present = [c for c in imu_cols if c in df.columns]

    def _wifi_mask(dfi: pd.DataFrame) -> pd.Series:
        return (dfi["label_X"].abs() <= xy_tol) & (dfi["label_Y"].abs() <= xy_tol)

    def _imu_mask(dfi: pd.DataFrame) -> pd.Series:
        y0, y1 = y_range
        return (dfi["label_X"].abs() <= xy_tol) & (dfi["label_Y"] >= y0 - xy_tol) & (dfi["label_Y"] <= y1 + xy_tol)

    ref_df = df[df["device"] == ref_device].copy()
    if ref_df.empty:
        return None

    ref_wifi = ref_df[_wifi_mask(ref_df)] if wifi_cols else pd.DataFrame()
    ref_imu = ref_df[_imu_mask(ref_df)] if imu_cols_present else pd.DataFrame()

    def _clip_series(v: pd.Series) -> pd.Series:
        if v.empty:
            return v
        if quantile_clip is None:
            return v
        try:
            q_low, q_high = float(quantile_clip[0]), float(quantile_clip[1])
        except (TypeError, ValueError):
            return v
        if not (0.0 <= q_low < q_high <= 1.0):
            return v
        lo = v.quantile(q_low)
        hi = v.quantile(q_high)
        return v.clip(lower=lo, upper=hi)

    def _robust_stats(series: pd.Series) -> Tuple[float, float, int]:
        v = pd.to_numeric(series, errors="coerce").dropna()
        if v.empty:
            return np.nan, np.nan, 0
        v = _clip_series(v)
        med = float(v.median())
        q25 = float(v.quantile(0.25))
        q75 = float(v.quantile(0.75))
        scale = q75 - q25
        if not np.isfinite(scale) or scale <= 0:
            scale = float(v.std(ddof=0))
        return med, scale, int(v.shape[0])

    def _shrink_weight(n: int) -> float:
        if shrink_k is None or shrink_k <= 0:
            return 1.0
        return float(n / (n + float(shrink_k)))

    def _clip_scale(val: float) -> float:
        if scale_clip is None:
            return val
        try:
            lo, hi = float(scale_clip[0]), float(scale_clip[1])
        except (TypeError, ValueError):
            return val
        if lo <= 0 or hi <= 0 or lo >= hi:
            return val
        return float(np.clip(val, lo, hi))

    ref_wifi_means = {}
    if not ref_wifi.empty:
        for c in wifi_cols:
            if c in ref_wifi.columns:
                ref_wifi_means[c] = float(pd.to_numeric(ref_wifi[c], errors="coerce").median())

    ref_imu_means = {}
    ref_imu_scales = {}
    if not ref_imu.empty:
        for c in imu_cols_present:
            med, scale, _ = _robust_stats(ref_imu[c])
            ref_imu_means[c] = float(med)
            ref_imu_scales[c] = float(scale)

    table: Dict[str, object] = {
        "reference_device": ref_device,
        "wifi_offset": {},
        "imu_scale": {},
        "imu_offset": {},
        "summary": {},
    }

    for dev in devices:
        dev_df = df[df["device"] == dev].copy()
        wifi_rows = int(_wifi_mask(dev_df).sum()) if not dev_df.empty else 0
        imu_rows = int(_imu_mask(dev_df).sum()) if not dev_df.empty else 0

        summary = {
            "device": dev,
            "wifi_rows": wifi_rows,
            "imu_rows": imu_rows,
            "wifi_cols": int(len(wifi_cols)),
            "imu_cols": int(len(imu_cols_present)),
            "status": "ok" if (wifi_rows >= min_rows or imu_rows >= min_rows) else "insufficient",
            "shrink_weight": _shrink_weight(max(wifi_rows, imu_rows)),
        }
        table["summary"][dev] = summary

        wifi_offsets = {}
        if wifi_cols and wifi_rows >= min_rows and ref_wifi_means:
            dev_wifi = dev_df[_wifi_mask(dev_df)]
            for c in wifi_cols:
                if c in dev_wifi.columns and c in ref_wifi_means:
                    dev_med, _, n_obs = _robust_stats(dev_wifi[c])
                    if np.isfinite(dev_med):
                        w = _shrink_weight(n_obs)
                        offset_raw = float(ref_wifi_means[c] - dev_med)
                        wifi_offsets[c] = float(w * offset_raw)
        table["wifi_offset"][dev] = wifi_offsets

        imu_scale = {}
        imu_offset = {}
        if imu_cols_present and imu_rows >= min_rows and ref_imu_means:
            dev_imu = dev_df[_imu_mask(dev_df)]
            for c in imu_cols_present:
                dev_med, dev_scale, n_obs = _robust_stats(dev_imu[c])
                ref_scale = ref_imu_scales.get(c, np.nan)
                ref_med = ref_imu_means.get(c, np.nan)
                if (
                    np.isfinite(dev_med)
                    and np.isfinite(dev_scale)
                    and np.isfinite(ref_scale)
                    and np.isfinite(ref_med)
                    and dev_scale > float(min_scale)
                    and ref_scale > 0
                ):
                    scale_raw = float(ref_scale / max(dev_scale, float(min_scale)))
                    scale_raw = _clip_scale(scale_raw)
                    w = _shrink_weight(n_obs)
                    scale = float((1.0 - w) + w * scale_raw)
                    scale = _clip_scale(scale)
                    offset_raw = float(ref_med - scale * dev_med)
                    offset = float(w * offset_raw)
                    imu_scale[c] = scale
                    imu_offset[c] = offset
        table["imu_scale"][dev] = imu_scale
        table["imu_offset"][dev] = imu_offset

    return table


def apply_device_calibration(
    df: pd.DataFrame,
    table: Dict[str, object] | None,
    *,
    imu_cols: Tuple[str, ...],
    wifi_prefixes: Tuple[str, ...],
) -> pd.DataFrame:
    """Apply per-device calibration offsets/scales to raw features."""

    if table is None or df is None or len(df) == 0:
        return df
    if "device" not in df.columns:
        return df

    out = df.copy()
    wifi_cols = get_wifi_columns(out, wifi_prefixes)
    imu_cols_present = [c for c in imu_cols if c in out.columns]

    wifi_offset = table.get("wifi_offset", {}) if isinstance(table, dict) else {}
    imu_scale = table.get("imu_scale", {}) if isinstance(table, dict) else {}
    imu_offset = table.get("imu_offset", {}) if isinstance(table, dict) else {}

    for dev in out["device"].dropna().unique():
        mask = out["device"] == dev
        if not mask.any():
            continue
        w_off = wifi_offset.get(dev, {})
        for c in wifi_cols:
            if c in w_off and c in out.columns:
                out.loc[mask, c] = pd.to_numeric(out.loc[mask, c], errors="coerce") + float(w_off[c])
        s_map = imu_scale.get(dev, {})
        b_map = imu_offset.get(dev, {})
        for c in imu_cols_present:
            if c in s_map and c in b_map and c in out.columns:
                out.loc[mask, c] = pd.to_numeric(out.loc[mask, c], errors="coerce") * float(s_map[c]) + float(b_map[c])

    return out


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

    # Robust IMU aggregate features: norms + local energy/jerk
    imu_extra_cols: List[str] = []
    accel_axes = [c for c in ("AccelX", "AccelY", "AccelZ") if c in df.columns]
    gyro_axes = [c for c in ("GyroX", "GyroY", "GyroZ") if c in df.columns]
    mag_axes = [c for c in ("MagnetoX", "MagnetoY", "MagnetoZ") if c in df.columns]

    if accel_axes:
        df["acc_norm"] = np.sqrt(np.sum(df[accel_axes].to_numpy(dtype=float) ** 2, axis=1))
        df["acc_jerk"] = (df["acc_norm"].diff() / dt_s).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        imu_extra_cols += ["acc_norm", "acc_jerk"]
    if gyro_axes:
        df["gyro_norm"] = np.sqrt(np.sum(df[gyro_axes].to_numpy(dtype=float) ** 2, axis=1))
        imu_extra_cols += ["gyro_norm"]
    if mag_axes:
        df["mag_norm"] = np.sqrt(np.sum(df[mag_axes].to_numpy(dtype=float) ** 2, axis=1))
        imu_extra_cols += ["mag_norm"]

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
        wifi_mat = df[topk_wifi_cols].to_numpy(dtype=float)
        df["wifi_mean"] = np.mean(wifi_mat, axis=1)
        df["wifi_std"] = np.std(wifi_mat, axis=1)
        df["wifi_max"] = np.max(wifi_mat, axis=1)
        df["wifi_min"] = np.min(wifi_mat, axis=1)
        wifi_extra_cols = ["wifi_mean", "wifi_std", "wifi_max", "wifi_min"]
    else:
        wifi_extra_cols = []

    feature_cols: List[str] = [time_col] + imu_cols_present
    feature_cols += deriv_cols
    feature_cols += imu_extra_cols
    feature_cols += roll_cols
    feature_cols += topk_wifi_cols
    feature_cols += wifi_extra_cols

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
        "imu_extra_cols": imu_extra_cols,
        "wifi_all_cols_count": len(wifi_all_cols),
        "wifi_topk_cols": topk_wifi_cols,
        "wifi_extra_cols": wifi_extra_cols,
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
