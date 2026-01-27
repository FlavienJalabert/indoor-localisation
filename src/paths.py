"""Centralized filesystem paths."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def outputs_dir() -> Path:
    return project_root() / "outputs"


def out_data_dir() -> Path:
    return outputs_dir() / "data"


def out_splits_dir() -> Path:
    return outputs_dir() / "splits"


def out_fe_dir() -> Path:
    return outputs_dir() / "fe"


def out_models_dir() -> Path:
    return outputs_dir() / "models"


def out_metrics_dir() -> Path:
    return outputs_dir() / "metrics"


def out_figures_dir() -> Path:
    return outputs_dir() / "figures"


def ensure_dirs() -> None:
    for p in [
        outputs_dir(),
        out_data_dir(),
        out_splits_dir(),
        out_fe_dir(),
        out_models_dir(),
        out_metrics_dir(),
        out_figures_dir(),
    ]:
        p.mkdir(parents=True, exist_ok=True)
