# Equipment Failure Analysis Methodology
    
## 1. Overview
This analysis defines the `failure_probability` target value for the REMHART Digital Twin.

## 2. Dynamic Baselines
- **Nominal Voltage ($V_{nom}$):** Calculated mean of phase voltages. Value: **126.49V**.
- **Rated Current ($I_{rated}$):** 98th Percentile of observed load. Value: **1.1360A**.
- **Ref:** IEEE Std 141-1993 (Maximum Demand Profiling).

## 3. The Law of Load Utilization
$$\text{Load %} = \left( \frac{\max(I_A, I_B, I_C)}{I_{rated}} \right) \times 100$$

## 4. Weighting Logic & References
| Stressor | Weight | Scientific Reference |
| :--- | :--- | :--- |
| **Thermal (Current)** | 45% | **IEEE Std C57.91-2011** (Thermal Aging Theory) |
| **Dielectric (Voltage)** | 35% | **IEC 60038** (Voltage Tolerances) |
| **Efficiency (PF)** | 20% | **IEEE Std 1459-2010** (Power Definitions) |
