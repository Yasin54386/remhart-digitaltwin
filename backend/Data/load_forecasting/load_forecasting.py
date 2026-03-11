import pandas as pd
import numpy as np
import os
import holidays

def generate_load_forecasting_dataset():
    # 1. Configuration
    input_file = "../master_data.xlsx"
    output_file = "load_forecasting_data.xlsx"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # 2. Load and Sort (Time-series integrity per IEEE Std 730)
    df = pd.read_excel(input_file, engine='openpyxl')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 1. TEMPORAL FEATURES (Social Seasonality)
    # Ref: Box-Jenkins Methodology
    # -------------------------------------------------------------------------
    df['hour_of_day'] = df['Timestamp'].dt.hour
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    df['day_of_month'] = df['Timestamp'].dt.day
    df['month'] = df['Timestamp'].dt.month

    # Cyclical Encoding to handle the 23-to-0 transition
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # -------------------------------------------------------------------------
    # 2. CALENDAR EVENTS (Deterministic Logic)
    # -------------------------------------------------------------------------
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Using Australian holidays for Northern Territory (Leanyer context)
    nt_holidays = holidays.Australia(prov='NT')
    df['is_public_holiday'] = df['Timestamp'].apply(lambda x: 1 if x in nt_holidays else 0)
    df['holiday_name'] = df['Timestamp'].apply(lambda x: nt_holidays.get(x) if x in nt_holidays else "None")

    # -------------------------------------------------------------------------
    # 3. HISTORICAL LAGS (Learning from P_T)
    # Ref: Autoregressive Persistence Theory
    # Lags based on 10-min sensor intervals: 1h=6 steps, 24h=144 steps, 168h=1008 steps.
    # -------------------------------------------------------------------------
    df['load_lag_1h'] = df['P_T'].shift(6)
    df['load_lag_24h'] = df['P_T'].shift(144)
    df['load_lag_168h'] = df['P_T'].shift(1008)

    # -------------------------------------------------------------------------
    # 4. TRAINING TARGET (The "Ground Truth")
    # Logic: This is the future value your model will try to predict. 
    # It is simply the P_T value from the very next row.
    # -------------------------------------------------------------------------
    df['target_load_next'] = df['P_T'].shift(-1)

    # 5. Final Processing: Remove rows where future/past data isn't available
    df.dropna(subset=['load_lag_168h', 'target_load_next'], inplace=True)

    # 6. Export
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"Dataset ready for ML training: {output_file}")

if __name__ == "__main__":
    generate_load_forecasting_dataset()