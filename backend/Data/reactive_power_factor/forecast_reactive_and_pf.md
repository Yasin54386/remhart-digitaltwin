# Forecast Reactive Power and Power Factor

## 1. Overview
This module facilitates **Power Flow Optimization** by predicting reactive power demand ($Q$) and power factor ($PF$) trends. It allows the **Digital Twin** to move from reactive mitigation to proactive optimization.

## 2. Logic & Feature Engineering
The model uses a **Supervised Learning** framework based on the following logic:
* **Temporal Demand Cycles**: Encodes **Social Seasonality** to account for periodic inductive load switching (e.g., motor starts at the beginning of work shifts).
* **Historical Persistence**: Uses multi-scale lags ($t-1h, t-24h$) to capture the autocorrelation of reactive flows.
* **Multivariate Input**: Analyzes the relationship between $P, Q, V,$ and $I$ to identify efficiency gaps.



## 3. Industry Standards & Value
* **IEEE 1459 Compliance**: Adheres to standards for measuring power quantities under non-sinusoidal conditions.
* **Voltage Stability**: Predictive $Q$ monitoring prevents "Voltage Sag" events by identifying precursors in reactive demand.
* **Loss Reduction**: By maintaining $PF$ near $1.0$, the model minimizes the "Unbalance Loss" and maximizes grid "Efficiency".