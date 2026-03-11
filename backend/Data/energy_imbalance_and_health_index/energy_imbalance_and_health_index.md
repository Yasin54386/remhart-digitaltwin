# Energy Imbalance and System Health Monitoring

## 1. Overview
The **Energy Imbalance and System Health Monitoring Model** provides a continuous verification of the grid’s physical integrity. It uses the mathematical constraints of 3-phase power to identify discrepancies between measured and theoretical values.

## 2. Scientific Logic & Standards
* **Physical Law Verification**: The model implements the fundamental formula $P = \sqrt{3} \times V \times I \times PF$ to cross-reference sensor data. Persistent "Energy Imbalance" is utilized as a primary indicator for Non-Technical Losses (NTL) or equipment degradation.
* **Phase Asymmetry Analysis**: Following **IEEE 1459**, the model monitors current unbalance percentages. Unbalance exceeding 15% is flagged as "Imbalance Loss," as it generates parasitic heat and reduces the efficiency of the measurement point.
* **System Health IndexING**: A health score (0-100) is derived from the **IEC 61000-4-30** measurement standard, quantifying the reliability of the system state.



## 3. Business Value
* **Imbalance Loss Detection**: Directly identifies when the 3-phase system is operating in an inefficient, unbalanced state, allowing for phase-balancing interventions.
* **Data Integrity**: Acts as a "watchdog" for sensor health, ensuring the Digital Twin's dashboards are displaying accurate, physically consistent information.
* **Thermal Protection**: High imbalance flags alert operators to potential motor or transformer overheating before catastrophic failure occurs.