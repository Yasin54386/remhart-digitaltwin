# Equipment Power Quality Index (E-PQI) Methodology

## 1. Overview
The Equipment Power Quality Index (E-PQI) is a comprehensive metric (0–100) designed for the **Digital Twin** to assess the suitability of electrical supply for specific assets. Unlike general grid-wide monitoring, the E-PQI quantifies how localized power anomalies impact the thermal health, dielectric integrity, and operational lifespan of equipment.

## 2. Algorithm and Processing Logic
The model utilizes a weighted Multi-Criteria Decision Analysis (MCDA) framework, mimicking the architecture of a Multi-Layer Perceptron (MLP) neural network. It processes five core electrical indicators to determine the final score:



### Weighted Factors:
| Stressor | Weight | Scientific Justification |
| :--- | :--- | :--- |
| **Voltage Stability** | 40% | Monitors RMS deviation from the dynamic nominal baseline (approx. 127V). |
| **Frequency Stability** | 30% | Evaluates adherence to the 60Hz fundamental frequency. |
| **Phase Balance** | 20% | Analyzes current unbalance using the 30% critical threshold limit. |
| **Efficiency (PF)** | 10% | Assesses reactive power compensation efficiency (FP_T). |

## 3. Training Dataset Logic & Grading
The E-PQI score is categorized into four distinct performance tiers based on deviation thresholds established through thousands of power quality scenarios:

| Grade | PQI Score | Voltage ($V$) | Frequency ($f$) | Phase Balance |
| :--- | :--- | :--- | :--- | :--- |
| **Excellent** | 90–100 | Within $\pm3\%$ | Within $\pm0.1Hz$ | Balanced phases |
| **Good** | 70–89 | Within $\pm5\%$ | Within $\pm0.2Hz$ | Minor imbalance |
| **Fair** | 50–69 | Within $\pm8\%$ | Within $\pm0.5Hz$ | Noticeable imbalance |
| **Poor** | Below 50 | $>8\%$ Deviation | $>0.5Hz$ Deviation | Severe imbalance |



## 4. Scientific Justification
This model is essential for grid operators to ensure reliable operation and asset longevity:
* **Holistic Assessment**: Provides a single metric summarizing complex interactions between voltage, frequency, harmonics, and phase balance.
* **SLA Compliance**: Monitors adherence to international power quality standards including **IEEE 1159** and **EN 50160**.
* **Root Cause Analysis**: Facilitates the identification of specific quality issues such as reactive power inefficiency or severe phase asymmetry.
* **Predictive Maintenance**: High correlation between PQI degradation and equipment failure rates allows this index to serve as a predictive indicator for the Digital Twin.

## 5. Justification for the 30% Unbalance Limit
Current unbalance is significantly more aggressive than voltage unbalance. A current unbalance of **30%** is treated as a critical failure point because it can double the internal temperature rise in equipment windings, leading to rapid insulation breakdown. This threshold aligns with **NEMA MG 1** standards for asset protection.