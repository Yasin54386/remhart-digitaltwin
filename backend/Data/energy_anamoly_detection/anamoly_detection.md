# Anamoly Detection in Energy Flow

## 1. Overview
The APQAD model transitions from traditional static threshold-based monitoring to **Unsupervised Machine Learning** to identify grid irregularities. Instead of monitoring parameters in isolation, this model learns the complex, hidden relationships between all electrical variables to establish a dynamic "Health Envelope".

## 2. Model Logic: Multivariate Pattern Recognition
The core logic of the model is based on the physical inter-dependency of ten distinct parameters:

### Feature Set:
* **Three-Phase Voltages**: `volt_A`, `volt_B`, `volt_C`.
* **Three-Phase Currents**: `I_A`, `I_B`, `I_C`.
* **Total Power Metrics**: Active Power (`P_T`), Reactive Power (`Q_T`), and Power Factor (`FP_T`).
* **Grid Stability**: Frequency (`Frec`).
* **Temporal Context**: Social Seasonality (Hour of Day).



### How the Model Learns:
* **Physical Constraints**: The model learns that in a healthy three-phase system, the voltages and currents follow **Kirchhoff’s and Ohm’s Laws**. For example, it understands that a drop in Power Factor ($PF$) must be mathematically balanced by a change in $P$ or $I$.
* **Isolation Forest Algorithm**: The model uses the **Isolation Forest** algorithm, which works by "isolating" data points that are few and different. Points that require fewer partitions to be isolated are assigned a higher **Anomaly Score**.
* **Contextual Awareness**: By incorporating **Social Seasonality**, the model distinguishes between "Night Normal" and "Day Normal". A high load that is normal at 5:00 PM on a weekday would be flagged as a major anomaly if it occurred at 3:00 AM.



## 3. Scientific Justification
The model's design is grounded in several international engineering and monitoring standards:

* **IEEE 1159 Compliance**: Adheres to the standard for characterizing long-term and transient power variations by analyzing the symmetry across all three phases.
* **Thermal Protection**: By monitoring all three phase voltages and currents, the model identifies **Phase Unbalance**. Large unbalances create negative sequence currents that can double the internal temperature of motor windings, leading to catastrophic failure.
* **Non-Technical Loss (NTL) Identification**: The model detects discrepancies in the $P/I$ relationship (e.g., power consumption that does not match the current draw), which are primary indicators of energy theft, bypasses, or sensor degradation.



## 4. Operational Value
* **Energy Loss Estimation**: Provides high-accuracy flags for "Energy Loss" by identifying major anomalies that suggest Non-Technical Losses (NTL).
* **Status Indicators**: Directly feeds into the overall system "Status," distinguishing between "Healthy," "Degraded," and "Anomalous" operational states.
* **Predictive Maintenance**: Enables the detection of "soft faults" (subtle behavioral shifts) weeks before they become "hard faults" or equipment failures.