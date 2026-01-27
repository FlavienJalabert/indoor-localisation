# Indoor Localization using WiFi and Inertial Sensors

> Master 2 ETAI Project - Polytech
> Indoor Localization / Machine Learning / Embedded Systems

---

## Description

This repository contains an **indoor localization** project based on **WiFi** signals and **inertial sensors**.  
The goal is to study the feasibility and limits of **data-driven** approaches for estimating a 2D position in indoor environments.

The project compares:

- **Pointwise (static)** models,
- **Sequential** models (LSTM / GRU),
- A **simple hybridation** between the two,

and analyzes their **cross-session and cross-device** robustness (ESP32 vs smartphone).

This work is part of an academic Master 2 project (ETAI) and focuses on a **rigorous methodological analysis** rather than state-of-the-art performance.

---

## Project Goals

- Estimate 2D position (X, Y) from WiFi and inertial sensor data.
- Compare static and sequential approaches.
- Study cross-device generalization.
- Identify structural limits of purely data-driven indoor localization.

---

## Data

The dataset includes multiple indoor trajectories recorded with:

- WiFi RSSI measurements,
- Inertial sensors (accelerometer, magnetometer, gyroscope),
- Absolute timestamps,
- Reference positions (X, Y labels).

Each trajectory is associated with:

- a device (`ESP32` or `Samsung`),
- a motion context (`motion`).

---

## Methodology

### Feature Engineering

- Selection and encoding of WiFi access points (top-k, presence, RSSI).
- Raw inertial signals with simple derivatives and rolling statistics.
- Encoding of contextual variables (`motion`).
- Controlled exclusion of `device` during cross-device tests.

### Models

- **Pointwise**: Random Forest, XGBoost, kNN
- **Sequential**: LSTM, GRU
- **Naive Hybrid**: linear combination of XGB / LSTM

### Evaluation

- Cross-device tests.
- Trajectory-based splits (no temporal leakage).
- Metrics: median, p90/p95, CDF of error.
- Visualization of trajectories and errors.

---

## Main Results

- Pointwise models show better spatial stability by RMSE.
- Sequential models improve temporal continuity but drift spatially (error accumulation).
- Naive hybridation shows potential but remains limited (too simple).
- Cross-device generalization is strongly affected by domain shift (offsets / bias).
- Without a map or explicit spatial constraints, performance hits a ceiling.

---

## Author

Master's student - Embedded Systems & Artificial Intelligence  
Flavien Jalabert
Polytech Nantes

---

## License

This project is provided for academic and educational purposes.
