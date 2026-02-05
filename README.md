# Indoor Localization with Temporal Deep Learning

Master 2 ETAI project, Polytech Nantes  
Topic: indoor positioning from WiFi + IMU time series

## Project framing

This repository studies 2D indoor localization under strict constraints:

- only the provided dataset is allowed
- the approach must be temporal
- deep learning must be used
- LSTM and GRU are mandatory
- the sequence heads follow a diamond topology

Diamond topology used in this project:

`n_features -> wider -> n_features -> narrower -> small -> outputs`

## Problem statement

Given a stream of sensor observations at time `t`:

- WiFi RSSI values
- accelerometer, gyroscope, magnetometer
- contextual metadata

predict the position target `(X_t, Y_t)` while learning motion dynamics, not only pointwise mapping.

## What this work evaluates

| Question | Why it matters |
|---|---|
| Can temporal DL models recover trajectory structure from this signal? | Core feasibility |
| How stable are predictions over time? | Motion consistency |
| What changes in cross-device transfer (ESP32 vs Samsung)? | Domain robustness |
| Are errors dominated by model design or data structure? | Practical limits |

## Data and assumptions

### Inputs

| Group | Columns (examples) |
|---|---|
| Time | `t_ms` |
| IMU | `Accel*`, `Gyro*`, `Magneto*` |
| WiFi | AP RSSI columns |
| Metadata | `device`, `motion`, `session_id` |
| Targets | `label_X`, `label_Y` and anchors when available |

### Working assumptions

- timestamps can be aligned to produce coherent training windows
- time continuity is meaningful inside segments
- some irregularity is expected (gaps, bursts, sparse anchors)
- evaluation must report coverage, not only error

## Method overview

1. Build a clean temporal base dataframe
   - timestamp normalization
   - label/feature alignment (`merge_asof`)
   - session segmentation
2. Build features
   - raw IMU + derivatives + rolling stats
   - WiFi top-k selection and aggregation
3. Build temporal windows
   - adaptive constraints and fallback strategy to preserve enough windows
   - optional controlled densification for coverage
4. Train deep sequence models
   - LSTM + diamond head
   - GRU + diamond head
5. Apply post-processing
   - kinematic guardrails
   - optional Kalman smoothing
6. Run intra-device and cross-device evaluation

## Models used in the report

| Type | Models |
|---|---|
| Temporal DL (main) | `LSTM_FE`, `GRU_FE` |
| Temporal DL + filtering | `LSTM_FE_KF`, `GRU_FE_KF` |
| Trajectory references | last-position oracle, constant-velocity oracle, constant-velocity rollout |

Note: non-DL tabular models may exist in code for diagnostics, but the graded core is the temporal DL pipeline.

## Comparisons and statistics

### Comparisons

- LSTM vs GRU
- raw vs Kalman-smoothed outputs
- intra-device vs cross-device
- sequence models vs trajectory baselines

### Metrics

| Category | Metrics |
|---|---|
| Spatial error | RMSE/MAE 2D |
| Distribution | median, p90, p95, p99 radial error |
| Axis behavior | `mae_x`, `mae_y`, `bias_x`, `bias_y` |
| Temporal reliability | cumulative error, valid prediction coverage |

### Plots generated

- trajectory overlays (`True` vs `Pred`)
- CDF error curves
- cumulative error curves
- per-axis error diagnostics
- cross-device trajectory views

## Reproducibility

Main notebook:

- `Report_Indoor_Localisation.ipynb`

Output folders:

- `outputs/metrics`
- `outputs/figures`
- `outputs/fe`
- `outputs/models`

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
