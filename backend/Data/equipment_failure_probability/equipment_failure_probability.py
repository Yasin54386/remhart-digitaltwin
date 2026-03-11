import pandas as pd
import numpy as np
import os

def generate_target_data_and_documentation():
    input_file = "master_data.xlsx"
    output_xlsx = "equipment_failure_probability_data.xlsx"
    output_md = "equipment_failure_analysis_methodology.md"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Ensure the merge script was run.")
        return

    # 2. Load Dataset
    df = pd.read_excel(input_file, engine='openpyxl')

    # --- DYNAMIC BASELINES (No Assumptions) ---
    # Nominal Voltage (V_nom): Grand mean of all phase voltage readings.
    v_cols = ['volt_A', 'volt_B', 'volt_C']
    v_nom = df[v_cols].mean().mean()

    # Rated Current (I_rated): 98th percentile of observed current.
    # REFERENCE: IEEE Std 141-1993 (Red Book) - Demand and Load Analysis.
    i_cols = ['I_A', 'I_B', 'I_C']
    i_rated = np.percentile(df[i_cols].dropna().values, 98)

    print(f"Established V_nom: {v_nom:.2f}V | I_rated: {i_rated:.4f}A")

    # --- FAILURE PROBABILITY CALCULATION ---
    
    # A. Thermal Stress (45%) - Ref: IEEE C57.91-2011
    # Logic: Aging rate is an exponential function of Joule Heating (I^2).
    max_i = df[i_cols].max(axis=1)
    i_score = np.clip(max_i / i_rated, 0, 1)

    # B. Dielectric Stress (35%) - Ref: IEC 60038
    # Logic: Insulation fatigue increases as voltage deviates from nominal 10% tolerance.
    v_avg = df[v_cols].mean(axis=1)
    v_dev = (v_avg - v_nom).abs()
    v_score = np.clip(v_dev / (v_nom * 0.10), 0, 1)

    # C. Efficiency Stress (20%) - Ref: IEEE 1459-2010
    # Logic: Low Power Factor increases reactive current, causing indirect thermal stress.
    pf_cols = ['FP_A', 'FP_B', 'FP_C']
    pf_avg = df[pf_cols].mean(axis=1)
    pf_score = pf_avg.apply(lambda x: 1.0 - x if x < 0.90 else 0.0)

    # --- FINAL WEIGHTED TARGET VALUE ---
    # Formula: (Thermal*0.45) + (Voltage*0.35) + (Efficiency*0.20)
    df['failure_probability'] = (i_score * 0.45) + (v_score * 0.35) + (pf_score * 0.20)
    df['failure_probability'] = df['failure_probability'].clip(0, 1).round(4)

    # Save Excel Result
    df.to_excel(output_xlsx, index=False, engine='openpyxl')
    
    # --- GENERATE METHODOLOGY DOCUMENTATION (.MD) ---
    md_content = f"""# Equipment Failure Analysis Methodology
    
## 1. Overview
This analysis defines the `failure_probability` target value for the REMHART Digital Twin.

## 2. Dynamic Baselines
- **Nominal Voltage ($V_{{nom}}$):** Calculated mean of phase voltages. Value: **{v_nom:.2f}V**.
- **Rated Current ($I_{{rated}}$):** 98th Percentile of observed load. Value: **{i_rated:.4f}A**.
- **Ref:** IEEE Std 141-1993 (Maximum Demand Profiling).

## 3. The Law of Load Utilization
$$\\text{{Load %}} = \\left( \\frac{{\\max(I_A, I_B, I_C)}}{{I_{{rated}}}} \\right) \\times 100$$

## 4. Weighting Logic & References
| Stressor | Weight | Scientific Reference |
| :--- | :--- | :--- |
| **Thermal (Current)** | 45% | **IEEE Std C57.91-2011** (Thermal Aging Theory) |
| **Dielectric (Voltage)** | 35% | **IEC 60038** (Voltage Tolerances) |
| **Efficiency (PF)** | 20% | **IEEE Std 1459-2010** (Power Definitions) |
"""
    with open(output_md, 'w') as f:
        f.write(md_content)

    print(f"Success! Generated {output_xlsx} and {output_md}")

if __name__ == "__main__":
    generate_target_data_and_documentation()