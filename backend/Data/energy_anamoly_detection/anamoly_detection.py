import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest

def generate_full_multivariate_anomaly_data():
    input_file = "../master_data.xlsx"
    output_file = "anomaly_detection_data.xlsx"

    if not os.path.exists(input_file): return

    df = pd.read_excel(input_file, engine='openpyxl')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # 1. FULL PARAMETER SELECTION (Multivariate Logic)
    # We include all three phase voltages and currents
    features = [
        'volt_A', 'volt_B', 'volt_C', 
        'I_A', 'I_B', 'I_C', 
        'P_T', 'Q_T', 'FP_T', 'Frec'
    ]
    
    # 2. TEMPORAL CONTEXT (Social Seasonality)
    # Essential for distinguishing between 'Night Normal' and 'Day Normal'
    df['hour'] = df['Timestamp'].dt.hour
    
    # Prepare data for the Learning Model
    X = df[features + ['hour']].fillna(0)

    # 3. UNSUPERVISED LEARNING (Isolation Forest)
    # The model learns the complex relationship between all phases.
    model = IsolationForest(n_estimators=100, contamination=0.03, random_state=42)
    
    # fit_predict creates the mathematical boundary of 'Normal' behavior
    df['anomaly_score'] = model.fit_predict(X)
    
    # 4. TARGET DERIVATION
    # -1 indicates the model found an 'unusual combination' of these 10 parameters.
    df['is_anomaly'] = np.where(df['anomaly_score'] == -1, 1, 0)

    df.to_excel(output_file, index=False)
    print(f"Dataset generated using Full Three-Phase Multivariate Logic.")

if __name__ == "__main__":
    generate_full_multivariate_anomaly_data()