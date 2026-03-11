import pandas as pd
import numpy as np
import os

def calculate_pqi_model():
    # 1. Configuration
    input_file = "master_data.xlsx"
    output_file = "equipment_pqi_score_data.xlsx"
    
    if not os.path.exists(input_file):
        print("Error: master_data.xlsx not found.")
        return

    # 2. Load Data
    df = pd.read_excel(input_file, engine='openpyxl')
    
    # Dynamic Nominal Voltage Baseline (Target for stability)
    v_cols = ['volt_A', 'volt_B', 'volt_C']
    v_nom = df[v_cols].mean().mean() 
    target_f = 60.0 # Standard Grid Frequency

    # ---------------------------------------------------------
    # 3. COMPONENT SCORING (Based on Neural Network Logic)
    # ---------------------------------------------------------

    # A. Voltage Stability Score (V_SI) - 40% Weight
    # Logic: V within ±3% is Excellent; ±8% is Fair (Ref: EN 50160)
    v_avg = df[v_cols].mean(axis=1)
    v_deviation_pct = (np.abs(v_avg - v_nom) / v_nom) * 100
    v_score = v_deviation_pct.apply(lambda x: 1.0 if x <= 3 else (0.7 if x <= 5 else (0.4 if x <= 8 else 0.1)))

    # B. Frequency Stability Score (F_SI) - 30% Weight
    # Logic: f within ±0.1Hz is Excellent (Ref: IEEE 1159)
    f_dev = np.abs(df['Frec'] - target_f)
    f_score = f_dev.apply(lambda x: 1.0 if x <= 0.1 else (0.7 if x <= 0.2 else (0.4 if x <= 0.5 else 0.1)))

    # C. Phase Balance Score (P_BI) - 20% Weight
    # Logic: Balanced phases = 1.0; Severe imbalance = 0.1 (Ref: NEMA MG 1)
    i_cols = ['I_A', 'I_B', 'I_C']
    i_mean = df[i_cols].mean(axis=1)
    unbalance_factor = (df[i_cols].max(axis=1) - df[i_cols].min(axis=1)) / i_mean.replace(0, 1)
    p_score = unbalance_factor.apply(lambda x: 1.0 if x <= 0.05 else (0.7 if x <= 0.15 else 0.1))

    # D. Efficiency (Reactive Power Compensation) - 10% Weight
    # Logic: Penalty for high reactive components (Ref: IEEE 1459)
    pf_score = df['FP_T'].apply(lambda x: 1.0 if x >= 0.95 else (0.5 if x >= 0.85 else 0.0))

    # ---------------------------------------------------------
    # 4. FINAL AGGREGATION (0-100)
    # ---------------------------------------------------------
    df['PQI_Score'] = ((v_score * 0.40) + (f_score * 0.30) + (p_score * 0.20) + (pf_score * 0.10)) * 100
    df['PQI_Score'] = df['PQI_Score'].round(0).astype(int)

    # Grading logic per Reference Screenshot
    def assign_grade(s):
        if s >= 90: return 'Excellent'
        elif s >= 70: return 'Good'
        elif s >= 50: return 'Fair'
        else: return 'Poor'

    df['PQI_Grade'] = df['PQI_Score'].apply(assign_grade)

    # 5. Export and Summary
    df.to_excel(output_file, index=False)
    print(f"PQI Model Generated. Current Grade: {df['PQI_Grade'].iloc[-1]}")

if __name__ == "__main__":
    calculate_pqi_model()