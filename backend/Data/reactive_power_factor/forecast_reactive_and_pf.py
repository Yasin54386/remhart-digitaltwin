import pandas as pd
import numpy as np
import os

def generate_reactive_optimization_dataset():
    input_file = "../master_data.xlsx"
    output_file = "reactive_and_pf_forecast_data.xlsx"

    if not os.path.exists(input_file): return

    # 1. Load and Sort (Time-Series Integrity per IEEE Std 730)
    df = pd.read_excel(input_file, engine='openpyxl')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 2. FEATURE ENGINEERING (Social Seasonality)
    # Logic: Reactive demand is highly correlated with industrial work shifts.
    # -------------------------------------------------------------------------
    df['hour_sin'] = np.sin(2 * np.pi * df['Timestamp'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['Timestamp'].dt.hour / 24)
    df['is_workday'] = df['Timestamp'].dt.dayofweek.apply(lambda x: 1 if x < 5 else 0)

    # -------------------------------------------------------------------------
    # 3. REACTIVE FLOW LAGS (Autoregressive Context)
    # Based on 10-min intervals: 1h=6 steps, 24h=144 steps.
    # -------------------------------------------------------------------------
    df['Q_lag_1h'] = df['Q_T'].shift(6)
    df['Q_lag_24h'] = df['Q_T'].shift(144)
    df['PF_lag_1h'] = df['FP_T'].shift(6)

    # -------------------------------------------------------------------------
    # 4. TARGET VARIABLES (Future Optimization Targets)
    # Prediction of future state to allow for proactive compensation.
    # -------------------------------------------------------------------------
    df['target_Q_next'] = df['Q_T'].shift(-1)
    df['target_PF_next'] = df['FP_T'].shift(-1)

    # 5. Clean and Export
    df.dropna(subset=['Q_lag_24h', 'target_Q_next'], inplace=True)
    df.to_excel(output_file, index=False)
    print(f"Reactive Optimization Dataset Created: {output_file}")

if __name__ == "__main__":
    generate_reactive_optimization_dataset()