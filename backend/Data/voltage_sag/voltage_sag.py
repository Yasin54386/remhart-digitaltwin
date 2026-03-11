import pandas as pd
import numpy as np
import os

def generate_voltage_sag_dataset():
    # 1. Path Configuration
    input_file = "master_data.xlsx"
    output_file = "voltage_sag_processed_data.xlsx"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # 2. Load Dataset
    df = pd.read_excel(input_file, engine='openpyxl')
    
    # 3. DYNAMIC NOMINAL VOLTAGE CALCULATION
    # Establishes the 127V baseline from your actual phase-to-neutral readings.
    v_cols = ['volt_A', 'volt_B', 'volt_C']
    v_nom = df[v_cols].mean().mean()

    # --- VOLTAGE SAG LOGIC ---
    # REFERENCE: IEEE 1159
    
    # A. Reactive Power Imbalance (Q/P Ratio)
    # Logic: Q/P < 0.3 is considered a normal operational reserve.
    df['Q_P_Ratio'] = (df['Q_T'] / df['P_T'].replace(0, 1)).abs()
    
    # B. Voltage Deviation Percentage
    # Calculates the magnitude of the dip relative to the calculated 127V nominal.
    v_avg = df[v_cols].mean(axis=1)
    df['v_dev_pct'] = (np.abs(v_avg - v_nom) / v_nom) * 100

    def calculate_sag_value(row):
        # NORMAL OPERATION: Stable voltage within ±5% and adequate Q reserves
        if row['v_dev_pct'] <= 5.0 and row['Q_P_Ratio'] < 0.3:
            return 0.0

        # SAG PRECURSORS: High reactive demand or weak source dynamics
        elif row['Q_P_Ratio'] >= 0.3 and row['v_dev_pct'] <= 10.0:
            return 0.5

        # LOAD-INDUCED SAGS: 10-30% drops (e.g., motor starting events)
        elif 10.0 < row['v_dev_pct'] <= 30.0:
            return 0.75

        # FAULT-INDUCED SAGS: 30-90% drops (e.g., fault conditions)
        elif row['v_dev_pct'] > 30.0:
            return 1.0

        return 0.25

    # Apply calculation to create the target column
    df['voltage_sag_probability'] = df.apply(calculate_sag_value, axis=1)

    # 4. Clean up and Export
    # We remove helper columns to keep the final dataset concise
    final_df = df.drop(columns=['v_dev_pct', 'Q_P_Ratio'])
    final_df.to_excel(output_file, index=False, engine='openpyxl')
    
    print(f"Dataset Processing Complete: {output_file}")
    print(f"Calculated System Nominal: {v_nom:.2f}V")

if __name__ == "__main__":
    generate_voltage_sag_dataset()