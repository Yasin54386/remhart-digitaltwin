import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest

def generate_explicit_voltage_anomaly_data():
    input_file = "../master_data.xlsx"
    output_file = "real_time_voltage_anomaly_data.xlsx"

    if not os.path.exists(input_file): return

    df = pd.read_excel(input_file, engine='openpyxl')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # --- EXPLICIT ANOMALY LOGIC ENGINEERING ---
    
    # 1. Phase Deviation Logic: Voltage Unbalance Factor (VUF%)
    # Logic: Significant deviations between phases are flagged.
    v_avg = df[['volt_A', 'volt_B', 'volt_C']].mean(axis=1)
    df['v_unbalance_max_dev'] = df[['volt_A', 'volt_B', 'volt_C']].sub(v_avg, axis=0).abs().max(axis=1)
    df['vuf_pct'] = (df['v_unbalance_max_dev'] / v_avg.replace(0, 1)) * 100

    # 2. Temporal Deviation Logic: Hour-over-Hour Stability
    # Logic: Identifying sudden shifts in voltage that don't match the time.
    df['hour'] = df['Timestamp'].dt.hour
    df['v_stability_delta'] = v_avg.diff().abs().fillna(0)

    # 3. SELECTING FEATURES FOR THE LEARNING MODEL
    # Now the model looks at the raw volts AND the calculated unbalance.
    features = ['volt_A', 'volt_B', 'volt_C', 'vuf_pct', 'v_stability_delta', 'hour']
    X = df[features].fillna(0)

    # 4. TRAIN THE ISOLATION FOREST
    # The 'contamination' flags the top 2% of deviations.
    model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    df['anomaly_score'] = model.fit_predict(X)
    
    # -1 is the label for Anomaly
    df['is_voltage_anomaly'] = np.where(df['anomaly_score'] == -1, 1, 0)

    df.to_excel(output_file, index=False)
    print(f"Anomalies found: {df['is_voltage_anomaly'].sum()} based on Explicit Phase & Temporal Logic.")

if __name__ == "__main__":
    generate_explicit_voltage_anomaly_data()