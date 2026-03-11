# Voltage Sag Prediction Methodology

## 1. Overview
This module computes a probability-based target for voltage sag events within the **REMHART Digital Twin** dataset. The calculation focuses on identifying expected magnitudes of sag events by analyzing reactive power dynamics and historical precursors.

## 2. Logic & Thresholds
The target value is derived from a systematic analysis of real-time measurements:

* **Normal Operation**: Defined as stable voltage within $\pm 5\%$ of the calculated nominal baseline (approx. $127V$) with adequate reactive reserves ($Q/P < 0.3$).
* **Sag Precursors**: Characterized by declining voltage trends and a high reactive power demand.
* **Fault-Induced Sags**: Simulated short-circuit or fault conditions causing a $30\% - 90\%$ voltage drop.
* **Load-Induced Sags**: Large motor starting events or transformer inrush leading to $10\% - 30\%$ voltage drops.

## 3. Scientific Justification
The logic is validated against **IEEE 1159** standards for sag characterization. Monitoring these parameters is essential for:
* **Preventing Process Interruptions**: Identifying potential sags before they impact industrial processes.
* **Protection Coordination**: Adjusting protective relay settings based on expected sag magnitudes.
* **Reactive Power Scheduling**: Deploying localized compensation (capacitors/SVCs) before sags fully develop.