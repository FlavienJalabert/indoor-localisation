"""Preprocessing helpers to build the base dataset."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Tuple

import pandas as pd

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
    return out


def densify_spatial_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate X/Y anchors to dense label_X/label_Y columns."""

    out = df.copy().reset_index(drop=True)
    if "X" not in out.columns or "Y" not in out.columns:
        return out

    n = len(out)
    if n == 0:
        return out

    out["label_X"] = pd.NA
    out["label_Y"] = pd.NA

    anchor_mask = out["X"].notna() & out["Y"].notna()
    anchor_idxs = list(out.index[anchor_mask])

    if len(anchor_idxs) == 0:
        return out

    if len(anchor_idxs) == 1:
        i0 = anchor_idxs[0]
        x0 = out.loc[i0, "X"]
        y0 = out.loc[i0, "Y"]
        out["label_X"] = x0
        out["label_Y"] = y0
        out = out.drop(columns=["X", "Y"])
        return out

    first = anchor_idxs[0]
    x_first = out.loc[first, "X"]
    y_first = out.loc[first, "Y"]
    out.loc[:first, "label_X"] = x_first
    out.loc[:first, "label_Y"] = y_first

    for i_start, i_end in zip(anchor_idxs[:-1], anchor_idxs[1:]):
        x1 = out.loc[i_start, "X"]
        y1 = out.loc[i_start, "Y"]
        x2 = out.loc[i_end, "X"]
        y2 = out.loc[i_end, "Y"]

        length = i_end - i_start
        if length <= 0:
            continue

        for i in range(i_start, i_end + 1):
            alpha = (i - i_start) / length
            xi = x1 + alpha * (x2 - x1)
            yi = y1 + alpha * (y2 - y1)
            out.loc[i, "label_X"] = xi
            out.loc[i, "label_Y"] = yi

    last = anchor_idxs[-1]
    x_last = out.loc[last, "X"]
    y_last = out.loc[last, "Y"]
    out.loc[last:, "label_X"] = x_last
    out.loc[last:, "label_Y"] = y_last

    out = out.drop(columns=["X", "Y"])
    out["label_X"] = out["label_X"].astype(float)
    out["label_Y"] = out["label_Y"].astype(float)
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


def build_final_dataset(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Apply metadata and label densification to all CSVs, then concat."""

    processed = []
    for name, df_item in dfs.items():
        df2 = add_meta_columns(df_item, name)
        df2 = densify_spatial_labels(df2)
        processed.append(df2)
    return pd.concat(processed, ignore_index=True, sort=True)


def trim_columns_by_nan(df: pd.DataFrame, max_nan_ratio: float) -> pd.DataFrame:
    """Drop columns with NaN ratio above max_nan_ratio."""

    null_ratio = df.isna().mean()
    cols_to_keep = null_ratio[null_ratio <= max_nan_ratio].index.tolist()
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


def make_base_dataframe(dfs: Dict[str, pd.DataFrame], *, max_nan_ratio: float) -> pd.DataFrame:
    """Build the cleaned base dataframe used by downstream steps."""

    df = build_final_dataset(dfs)
    df = trim_columns_by_nan(df, max_nan_ratio=max_nan_ratio)

    if {"label_X", "label_Y"}.issubset(df.columns):
        df = df.dropna(subset=["label_X", "label_Y"]).copy()

    if "Timestamp" in df.columns:
        df = df.copy()
        df["t_ms"] = timestamp_to_ms(df["Timestamp"])

    df = drop_leakage_and_useless_cols(df, DEFAULT_COLS_TO_DROP)
    df = impute_magneto_median(df)

    required = {"device", "motion", "t_ms", "label_X", "label_Y"}
    missing = required.difference(df.columns)
    if missing:
        LOGGER.warning("Base dataframe missing required columns: %s", sorted(missing))

    return df
