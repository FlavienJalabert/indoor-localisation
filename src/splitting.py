"""Train/val/test splitting utilities."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def add_session_id(
    df: pd.DataFrame, cols: Tuple[str, str] = ("device", "motion"), name: str = "session_id"
) -> pd.DataFrame:
    """Add a session identifier by concatenating device/motion columns."""

    out = df.copy()
    out[name] = out[cols[0]].astype(str) + "__" + out[cols[1]].astype(str)
    return out


def group_split(
    df: pd.DataFrame, group_col: str, test_size: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Group-aware train/test split."""

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df[group_col]))
    return (
        df.iloc[train_idx].copy(),
        df.iloc[test_idx].copy(),
    )


def group_split_train_val(
    df_train: pd.DataFrame, group_col: str, val_size: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Group-aware train/val split from a training dataframe."""

    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr_idx, val_idx = next(gss.split(df_train, groups=df_train[group_col]))
    return (
        df_train.iloc[tr_idx].copy(),
        df_train.iloc[val_idx].copy(),
    )
