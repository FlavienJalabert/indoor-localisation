"""Preprocessing helpers to build the base dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
import numpy as np

LOGGER = logging.getLogger(__name__)


DEFAULT_COLS_TO_DROP = (
    "Index",
    "Timestamp",
    "vX",
    "Orientation",
    "vY",
    "RefP",
)


def infer_device_and_motion(filename: str) -> Tuple[str, str]:
    """Infer device and motion from filename."""

    name = filename.lower()
    device = "esp32" if "esp32" in name else ("samsung" if "samsung" in name else "unknown")
    if "horizontal" in name:
        motion = "horizontal"
    elif "vertical" in name:
        motion = "vertical"
    elif "square" in name:
        motion = "square"
    elif "combined" in name or "combine" in name:
        motion = "combined"
    else:
        motion = "unknown"
    return device, motion


def add_meta_columns(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Add device/motion columns inferred from filename."""

    device, motion = infer_device_and_motion(filename)
    out = df.copy()
    out["device"] = device
    out["motion"] = motion
    out["session_file"] = Path(filename).stem
    return out


def _aggregate_by_time(
    df: pd.DataFrame, *, session_col: str, time_col: str
) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    agg: Dict[str, str] = {}
    for c in df.columns:
        if c == time_col or c == session_col:
            agg[c] = "first"
        elif c in numeric_cols:
            agg[c] = "mean"
        else:
            agg[c] = "last"
    return (
        df.groupby([session_col, time_col], as_index=False, sort=False)
        .agg(agg)
        .sort_values([session_col, time_col], kind="mergesort")
        .reset_index(drop=True)
    )


def _ensure_session_id(df: pd.DataFrame, session_col: str = "session_id") -> pd.DataFrame:
    if session_col in df.columns:
        return df
    out = df.copy()
    if "session_file" in out.columns:
        out[session_col] = out["session_file"].astype(str)
    elif {"device", "motion"}.issubset(out.columns):
        out[session_col] = out["device"].astype(str) + "__" + out["motion"].astype(str)
    else:
        out[session_col] = "session_0"
    return out


def align_labels_asof(
    df: pd.DataFrame,
    *,
    time_col: str = "t_ms",
    session_col: str = "session_id",
    direction: str = "nearest",
    tolerance_ms: float | None = 500.0,
) -> pd.DataFrame:
    """Align sparse X/Y anchors to each row using merge_asof per session."""

    if "X" not in df.columns or "Y" not in df.columns or time_col not in df.columns:
        return df

    out = _ensure_session_id(df, session_col=session_col)

    out = out.sort_values([session_col, time_col], kind="mergesort").reset_index(drop=True)
    out = _aggregate_by_time(out, session_col=session_col, time_col=time_col)

    labels = out[[session_col, time_col, "X", "Y"]].dropna(subset=["X", "Y"]).copy()
    if labels.empty:
        return out

    feats = out.drop(columns=["X", "Y"])
    # Ensure tolerance dtype compatible with time_col dtype
    tol = tolerance_ms
    if tol is not None:
        time_dtype = out[time_col].dtype
        if np.issubdtype(time_dtype, np.integer):
            tol = int(tol)
        else:
            tol = float(tol)

    merged_parts = []
    for sid, g in feats.groupby(session_col, sort=False):
        anchors = labels[labels[session_col] == sid].drop(columns=[session_col])
        if anchors.empty:
            tmp = g.copy()
            tmp["label_X"] = pd.NA
            tmp["label_Y"] = pd.NA
            merged_parts.append(tmp)
            continue
        aligned = pd.merge_asof(
            g.sort_values(time_col, kind="mergesort"),
            anchors.sort_values(time_col, kind="mergesort"),
            on=time_col,
            direction=direction,
            tolerance=tol,
            allow_exact_matches=True,
        )
        aligned = aligned.rename(columns={"X": "label_X", "Y": "label_Y"})
        merged_parts.append(aligned)

    out = pd.concat(merged_parts, ignore_index=True)
    return out


def timestamp_to_ms(ts: pd.Series) -> pd.Series:
    """Convert Timestamp HH:MM:SS:xx to milliseconds."""

    parts = ts.astype(str).str.split(":", expand=True)
    if parts.shape[1] != 4:
        raise ValueError("Timestamp must be in format HH:MM:SS:xx")
    hh = pd.to_numeric(parts[0], errors="coerce")
    mm = pd.to_numeric(parts[1], errors="coerce")
    ss = pd.to_numeric(parts[2], errors="coerce")
    xx = pd.to_numeric(parts[3], errors="coerce")
    return ((hh * 3600 + mm * 60 + ss) * 1000) + (xx * 10)


def build_final_dataset(
    dfs: Dict[str, pd.DataFrame],
    *,
    merge_direction: str = "nearest",
    merge_tolerance_ms: float | None = 500.0,
) -> pd.DataFrame:
    """Apply metadata and label alignment to all CSVs, then concat."""

    processed = []
    for name, df_item in dfs.items():
        df2 = add_meta_columns(df_item, name)
        if "Timestamp" in df2.columns and "t_ms" not in df2.columns:
            df2 = df2.copy()
            df2["t_ms"] = timestamp_to_ms(df2["Timestamp"])
        df2 = align_labels_asof(
            df2,
            time_col="t_ms",
            direction=merge_direction,
            tolerance_ms=merge_tolerance_ms,
        )
        processed.append(df2)
    return pd.concat(processed, ignore_index=True, sort=True)


def trim_columns_by_nan(df: pd.DataFrame, max_nan_ratio: float) -> pd.DataFrame:
    """Drop columns with NaN ratio above max_nan_ratio."""

    null_ratio = df.isna().mean()
    protected_cols = {
        "session_id",
        "session_file",
        "device",
        "motion",
        "t_ms",
        "dt_ms",
        "label_X",
        "label_Y",
        "segment_id",
    }
    cols_to_keep = [c for c in df.columns if (c in protected_cols) or (null_ratio.get(c, 1.0) <= max_nan_ratio)]
    return df[cols_to_keep].copy()


def drop_leakage_and_useless_cols(df: pd.DataFrame, cols_to_drop: Iterable[str]) -> pd.DataFrame:
    """Drop known leakage/useless columns if present."""

    drop_list = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=drop_list)


def impute_magneto_median(
    df: pd.DataFrame, cols: Tuple[str, str, str] = ("MagnetoX", "MagnetoY", "MagnetoZ")
) -> pd.DataFrame:
    """Impute magnetometer columns with their median."""

    out = df.copy()
    for c in cols:
        if c in out.columns and out[c].isnull().any():
            out[c] = out[c].fillna(out[c].median())
    return out


def make_base_dataframe(
    dfs: Dict[str, pd.DataFrame],
    *,
    max_nan_ratio: float,
    merge_direction: str = "nearest",
    merge_tolerance_ms: float | None = 500,
    dt_max_gap_ms: float | None = 1000,
    gap_thr_ms: float | None = 1000,
) -> pd.DataFrame:
    """Build the cleaned base dataframe used by downstream steps."""

    df = build_final_dataset(
        dfs,
        merge_direction=merge_direction,
        merge_tolerance_ms=merge_tolerance_ms,
    )
    df = trim_columns_by_nan(df, max_nan_ratio=max_nan_ratio)

    if {"label_X", "label_Y"}.issubset(df.columns):
        df = df.dropna(subset=["label_X", "label_Y"]).copy()

    if "Timestamp" in df.columns and "t_ms" not in df.columns:
        df = df.copy()
        df["t_ms"] = timestamp_to_ms(df["Timestamp"])

    df = drop_leakage_and_useless_cols(df, DEFAULT_COLS_TO_DROP)
    df = impute_magneto_median(df)

    if "t_ms" in df.columns:
        df = _ensure_session_id(df, session_col="session_id")
        df = df.sort_values(["session_id", "t_ms"], kind="mergesort").reset_index(drop=True)
        df = df.copy()
        df["dt_ms"] = df.groupby(["session_id"])["t_ms"].diff()
        df = df[df["dt_ms"].isna() | (df["dt_ms"] > 0)].copy()
        if dt_max_gap_ms is not None:
            df = df[df["dt_ms"].isna() | (df["dt_ms"] <= dt_max_gap_ms)].copy()
        if gap_thr_ms is not None:
            df = df.copy()
            df["__new_segment"] = df["dt_ms"].isna() | (df["dt_ms"] > gap_thr_ms)
            df["segment_id"] = (
                df.groupby(["session_id"])["__new_segment"].cumsum().astype(int)
            )
            df = df.drop(columns=["__new_segment"])
        df = df.reset_index(drop=True)

    required = {"session_id", "device", "motion", "t_ms", "label_X", "label_Y"}
    missing = required.difference(df.columns)
    if missing:
        LOGGER.warning("Base dataframe missing required columns: %s", sorted(missing))

    return df
