"""Configuration and seed helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass
class Config:
    """Central configuration for the indoor localization pipeline."""

    zip_url: str = (
        "https://uncloud.univ-nantes.fr/public.php/dav/files/fkkT27xoRkNRmsr/?accept=zip"
    )
    cache_dir_raw: Path = Path("outputs/data/raw")

    max_nan_ratio: float = 0.56
    wifi_prefixes: Tuple[str, ...] = (
        "eduroam",
        "tp-link",
        "clickshare",
        "telephone",
        "elb",
        "direct",
        "_be",
    )
    wifi_topk: int = 10
    wifi_min_presence: float = 0.05
    rssi_missing: float = -100.0

    # Feature engineering defaults
    time_col: str = "t_ms"
    imu_cols: Tuple[str, ...] = (
        "AccelX",
        "AccelY",
        "AccelZ",
        "GyroX",
        "GyroY",
        "GyroZ",
        "MagnetoX",
        "MagnetoY",
        "MagnetoZ",
    )
    rolling_group_cols: Tuple[str, ...] = ("session_id",)
    rolling_imu_cols: Tuple[str, ...] = (
        "AccelX",
        "AccelY",
        "AccelZ",
        "GyroX",
        "GyroY",
        "GyroZ",
    )
    rolling_stats: Tuple[str, ...] = ("mean", "var", "min", "max")
    add_rolling: bool = True
    add_diff: bool = True
    add_dt_derivative: bool = True
    eps_dt_ms: float = 1.0
    fill_numeric_with: str = "median"
    merge_direction: str = "nearest"
    merge_tolerance_ms: float | None = 500.0
    dt_max_gap_ms: float | None = 1000.0
    gap_thr_ms: float = 1000.0
    seq_max_window_ms: float | None = None
    use_kalman_postproc: bool = True
    use_pred_guardrails: bool = True
    kalman_process_var: float = 1e-3
    kalman_meas_var: float = 1e-1
    baseline_max_speed_mps: float = 2.5

    rolling_window_size: int = 10
    window_size: int = 8
    seq_max_dt_ms: float | None = None
    epochs: int = 100
    patience: int = 20
    batch_size: int = 128
    lr: float = 5e-4
    seq_use_pca: bool = False
    seq_pca_n_components: int = 20
    seq_pca_topk_corr: int = 20
    top_k: int = 20
    seq_use_topk_corr: bool = False
    seq_topk_corr_k: int = 20
    seq_target_mode: str = "delta"
    run_seq_pca_experiment: bool = False
    seq_use_anchor_points: bool = False
    seq_include_time_feature: bool = False
    seq_ignore_time_checks: bool = True
    seq_densify_step_ms: int | None = 160
    seq_densify_max_factor: float = 4.0
    seq_min_test_windows: int = 80
    seq_min_coverage: float = 0.25
    seq_min_window_size: int = 4

    test_size: float = 0.3
    val_size: float = 0.2
    random_seed: int = 1242

    thresholds: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)

    cols_to_drop: Tuple[str, ...] = (
        "Index",
        "Timestamp",
        "vX",
        "Orientation",
        "vY",
        "RefP",
    )

    xgb_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 600,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        }
    )
    rf_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 500,
            "max_depth": None,
            "random_state": 42,
            "n_jobs": -1,
        }
    )
    knn_params: Dict[str, Any] = field(default_factory=lambda: {"n_neighbors": 5, "n_jobs": -1})

    lstm_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "hidden_dim": 64,
            "num_layers": 1,
            "dropout_lstm": 0.1,
            "dropout_fc": 0.05,
        }
    )
    gru_params: Dict[str, Any] = field(
        default_factory=lambda: {"hidden_dim": 32, "num_layers": 1, "dropout_gru": 0.25}
    )
    gru_lr: float = 2e-4
    gru_patience: int = 12


def set_global_seed(seed: int) -> None:
    """Seed numpy/random/torch for reproducibility."""

    import random

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
