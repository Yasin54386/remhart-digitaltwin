import pandas as pd
import numpy as np
import os

def calculate_overload_and_risk():
    # 1. Define Paths
    input_file = "master_data.xlsx"
    output_file = "overload_risk_classification_data.xlsx"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # 2. Load Master Dataset
    df = pd.read_excel(input_file, engine='openpyxl')

    # 3. DATA CLEANING: Drop rows where current values are missing or zero
    # This prevents the 'nan' result in percentile calculations.
    i_cols = ['I_A', 'I_B', 'I_C']
    df_clean = df.dropna(subset=i_cols)
    df_clean = df_clean[(df_clean[i_cols] > 0).all(axis=1)]

    if df_clean.empty:
        print("Error: No valid numeric current data found for calculation.")
        return

    # -------------------------------------------------------------------------
    # 4. ESTABLISH RATED CAPACITY (I_rated)
    # REFERENCE: IEEE Std 141-1993 (Red Book) - Demand and Load Analysis.
    # LOGIC: 98th Percentile represents the Maximum Normal Demand baseline.
    # -------------------------------------------------------------------------
    i_rated = np.percentile(df_clean[i_cols].values, 98)

    # -------------------------------------------------------------------------
    # 5. THE LAW OF LOAD UTILIZATION (Load %)
    # REFERENCE: IEEE Std C57.91-2011 (Thermal aging of transformers).
    # FORMULA: (Max(I_A, I_B, I_C) / I_rated) * 100
    # -------------------------------------------------------------------------
    df['max_current'] = df[i_cols].max(axis=1)
    df['current_load_pct'] = (df['max_current'] / i_rated) * 100

    # -------------------------------------------------------------------------
    # 6. PEAK PHASE IDENTIFICATION
    # REFERENCE: IEEE Std 1459-2010 (Measurement of Power Quantities).
    # -------------------------------------------------------------------------
    df['peak_phase'] = df[i_cols].idxmax(axis=1).str.replace('I_', '')

    # -------------------------------------------------------------------------
    # 7. OVERLOAD RISK CLASSIFICATION
    # REFERENCE: NFPA 70 (NEC) - The 80% Continuous Load Rule.
    # -------------------------------------------------------------------------
    def classify_risk(load_pct):
        if pd.isna(load_pct): return 'Unknown'
        if load_pct < 80: return 'Low'      # Safe (NEC 80% Margin)
        elif load_pct < 90: return 'Medium' # Approaching thermal limit
        else: return 'High'                 # Critical Overload
    
    df['overload_risk'] = df['current_load_pct'].apply(classify_risk)

    # -------------------------------------------------------------------------
    # 8. EXPORT DATA
    # -------------------------------------------------------------------------
    df['current_load_pct'] = df['current_load_pct'].round(2)
    df['calculated_i_rated'] = round(i_rated, 4)
    
    df.to_excel(output_file, index=False, engine='openpyxl')
    
    print(f"Analysis Successful.")
    print(f"Calculated Equipment Rated Limit (98th Pctl): {i_rated:.3f} A")

if __name__ == "__main__":
    calculate_overload_and_risk()