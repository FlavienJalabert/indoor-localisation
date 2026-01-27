"""Post-processing filters for trajectories."""

from __future__ import annotations

import numpy as np


def kalman_filter_2d(
    y_obs: np.ndarray,
    *,
    t_ms: np.ndarray | None = None,
    process_var: float = 1e-3,
    meas_var: float = 1e-1,
    init_pos: np.ndarray | None = None,
    init_vel: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Apply a constant-velocity Kalman filter to 2D positions.

    Args:
        y_obs: (N,2) array of observed positions.
        t_ms: optional timestamps (ms) to derive variable dt. If None, dt=1.
        process_var: acceleration noise variance.
        meas_var: measurement noise variance.
        init_pos: optional initial position. If None, uses first finite obs.
        init_vel: initial velocity (vx, vy).

    Returns:
        (N,2) array of filtered positions.
    """

    y_obs = np.asarray(y_obs, dtype=float)
    if y_obs.ndim != 2 or y_obs.shape[1] != 2:
        raise ValueError("y_obs must be (N,2)")

    n = y_obs.shape[0]
    if n == 0:
        return y_obs.copy()

    if t_ms is None:
        dt_seq = np.ones(n - 1, dtype=float)
    else:
        t_ms = np.asarray(t_ms, dtype=float)
        if t_ms.shape[0] != n:
            raise ValueError("t_ms length must match y_obs")
        dt_seq = np.diff(t_ms) / 1000.0
        # fallback if non-positive
        med = np.nanmedian(dt_seq) if dt_seq.size else 1.0
        if not np.isfinite(med) or med <= 0:
            med = 1.0
        dt_seq = np.where(dt_seq <= 0, med, dt_seq)

    # state: [x, y, vx, vy]
    x = np.zeros(4, dtype=float)

    if init_pos is None:
        finite_mask = np.isfinite(y_obs).all(axis=1)
        if finite_mask.any():
            init_pos = y_obs[finite_mask][0]
        else:
            init_pos = np.zeros(2, dtype=float)
    x[0:2] = np.asarray(init_pos, dtype=float)
    x[2:] = np.asarray(init_vel, dtype=float)

    P = np.eye(4, dtype=float) * 10.0
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
    R = np.eye(2, dtype=float) * float(meas_var)
    I = np.eye(4, dtype=float)

    out = np.zeros((n, 2), dtype=float)

    for i in range(n):
        if i > 0:
            dt = float(dt_seq[i - 1])
            F = np.array(
                [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                dtype=float,
            )
            q = float(process_var)
            dt2 = dt * dt
            dt3 = dt2 * dt
            dt4 = dt3 * dt
            Q = q * np.array(
                [[dt4 / 4.0, 0.0, dt3 / 2.0, 0.0], [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                 [dt3 / 2.0, 0.0, dt2, 0.0], [0.0, dt3 / 2.0, 0.0, dt2]],
                dtype=float,
            )
            x = F @ x
            P = F @ P @ F.T + Q

        z = y_obs[i]
        if np.isfinite(z).all():
            y = z - (H @ x)
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (I - K @ H) @ P

        out[i] = x[0:2]

    return out
