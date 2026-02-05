# Indoor Localization with Temporal Deep Learning

Academic project, Polytech Nantes  
Topic: indoor positioning from WiFi + IMU time series

## Study scope

This repository investigates 2D indoor localization under the following constraints:

- only the provided dataset is used
- the approach must be temporal
- deep learning must be used
- both LSTM and GRU models are required
- sequence heads follow a diamond topology

Diamond topology used in this work:

`n_features -> wider -> n_features -> narrower -> small -> outputs`

## Problem statement

Given a stream of sensor observations at time `t`:

- WiFi RSSI values
- accelerometer, gyroscope, magnetometer signals
- contextual metadata

predict the position target `(X_t, Y_t)` while modeling trajectory dynamics rather than pointwise mapping only.

## Domain-shift normalization protocol (reference)

### Problem

Cross-device transfer (ESP32 ↔ Samsung) introduces a distribution shift that motivates a normalization protocol.
The goal is to correct this shift by applying a standardized normalization during feature engineering, based on a device correspondence table.

### Protocol

- **Reference device:** ESP32 is the canonical domain (models are trained on ESP32).
- **Calibration anchors:** two reference actions are used to derive coefficients:
  - **WiFi RSSI:** a static calibration at position (0,0) provides per‑AP offsets.
  - **IMU:** a short linear calibration segment from (0,0) to (0,1) provides scale/offset alignment for inertial features.
- **Mapping:** for each device `d`, learn a feature‑wise affine mapping so that
  `features_d * A_d + b_d ≈ features_esp` (per feature group, applied during FE).
  - **Robust estimation:** offsets/scales use median + IQR (with quantile clipping) and are shrunk toward identity
    when the calibration sample is small, to avoid unstable corrections.
- **Final standardization (optional):** after feature engineering, numeric features can be z‑scored using
   training statistics (per‑device when available, otherwise global).

### Validation criterion

At the reference point (0,0), all normalized features must match the ESP32 reference within a fixed tolerance.
The same calibration segment should produce comparable IMU feature statistics after normalization.

### Operational requirement

In real‑world use, a brief **manual calibration step at (0,0)** is required before using the model on a new device.

### Post-processing alignment (optional)

If cross-device predictions show a mirrored trajectory, a lightweight sign correction can be applied:
when trajectories violate the expected frame (e.g., negative X or Y), the post-processing flips the **X**
sign only. This is label-free and corrects global inversions without altering local turns. A calibration‑segment
alignment mode remains available, but the default is the bounds‑based X‑sign correction for robustness.

## Research questions

| Question | Why it matters |
| --- | --- |
| Can temporal deep models recover trajectory structure from this signal? | Core feasibility |
| How stable are predictions over time? | Motion consistency |
| What changes under cross-device transfer (ESP32 vs Samsung)? | Domain robustness |
| What factors drive performance differences? | Practical interpretation |

## Data and assumptions

### Inputs

| Group | Columns (examples) |
| --- | --- |
| Time | `t_ms` |
| IMU | `Accel*`, `Gyro*`, `Magneto*` |
| WiFi | AP RSSI columns |
| Metadata | `device`, `motion`, `session_id` |
| Targets | `label_X`, `label_Y` |

### Working assumptions

- timestamps can be aligned into coherent training windows
- temporal continuity is informative inside segments
- irregularity is expected (gaps, bursts, sparse anchors)
- evaluation must report coverage in addition to error

## Method overview

1. Build a clean temporal base dataframe
   - timestamp normalization
   - label/feature alignment (`merge_asof`)
   - session segmentation
2. Build features
   - raw IMU + derivatives + rolling statistics
   - WiFi top-k selection and aggregation
   - optional final feature standardization (z-score on train stats)
3. Build temporal windows
   - adaptive constraints and fallback strategy to preserve coverage
   - optional controlled densification
4. Train deep sequence models
   - LSTM + diamond head
   - GRU + diamond head
5. Apply post-processing
   - kinematic guardrails
   - optional Kalman smoothing
   - optional trajectory alignment using the calibration segment (rigid 2D flip/rotation)
6. Evaluate intra-device and cross-device settings

## Models reported

| Type | Models |
| --- | --- |
| Temporal DL (main) | `LSTM_FE`, `GRU_FE` |
| Temporal DL + filtering | `LSTM_FE_KF`, `GRU_FE_KF` |
| Trajectory references | last-position oracle, constant-velocity oracle, constant-velocity rollout |

## Evaluation design

### Comparisons

- LSTM vs GRU
- raw predictions vs Kalman-smoothed predictions
- intra-device vs cross-device performance
- sequential models vs trajectory baselines

### Metrics

| Category | Metrics |
| --- | --- |
| Spatial error | RMSE/MAE 2D |
| Distribution | median, p90, p95, p99 radial error |
| Axis behavior | `mae_x`, `mae_y`, `bias_x`, `bias_y` |
| Reliability | cumulative error and valid-prediction coverage |

### Figures

- trajectory overlays (`True` vs `Pred`)
- error CDF curves
- cumulative error curves
- axis-wise error diagnostics
- cross-device trajectory views

## Reproducibility

Main notebook:

- `Report_Indoor_Localisation.ipynb`

Output folders:

- `outputs/metrics`
- `outputs/figures`
- `outputs/fe`
- `outputs/models`
- `outputs/reports`

## Setup

### Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 \
    --index-url https://pypi.org/simple \
    -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install jupyter
jupyter notebook
```

### Windows

```powershell
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 `
    --index-url https://pypi.org/simple `
    -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
python -m pip install jupyter
jupyter notebook
```

## Author

Flavien Jalabert  
Master ETAI, Polytech Nantes

## License

Academic and educational use.
