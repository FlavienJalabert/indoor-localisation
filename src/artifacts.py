"""Artifacts I/O helpers and path conventions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

import paths


def artifact_path(kind: str, name: str, run_id: str, ext: str) -> Path:
    """Build a standardized artifact path under outputs/{kind}."""

    kind = kind.strip().lower()
    base = {
        "data": paths.out_data_dir(),
        "splits": paths.out_splits_dir(),
        "fe": paths.out_fe_dir(),
        "models": paths.out_models_dir(),
        "metrics": paths.out_metrics_dir(),
        "figures": paths.out_figures_dir(),
    }.get(kind)
    if base is None:
        raise ValueError(f"Unknown artifact kind: {kind}")
    filename = f"{name}__{run_id}.{ext.lstrip('.')}"
    return base / filename


def save_df(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to csv based on suffix."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError("Unsupported DataFrame format. Use .csv")


def load_df(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Unsupported DataFrame format. Use .csv")


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def save_joblib(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: Path) -> Any:
    return joblib.load(path)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    obj = _jsonify(obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def exists(path: Path) -> bool:
    return path.exists()

def _jsonify(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj
