import pandas as pd
import numpy as np
import os

def generate_health_monitoring_dataset():
    input_file = "../master_data.xlsx"
    output_file = "energy_imbalance_and_health_index_data.xlsx"

    if not os.path.exists(input_file):
        return

    # 1. Load Data
    df = pd.read_excel(input_file, engine='openpyxl')
    
    # 2. PHYSICAL RELATIONSHIP CALCULATIONS
    # Calculating the expected P based on the Square Root of 3 Law
    v_avg = df[['volt_A', 'volt_B', 'volt_C']].mean(axis=1)
    i_avg = df[['I_A', 'I_B', 'I_C']].mean(axis=1)
    
    # Theoretical P = sqrt(3) * V_avg * I_avg * PF //Electrical Fundamental formula
    df['calculated_P'] = (np.sqrt(3) * v_avg * i_avg * df['FP_T']) / 1000 # converting to kW
    
    # 3. ENERGY IMBALANCE & LOSS LOGIC
    # Energy Imbalance (P_Measured - P_Calculated)
    df['energy_imbalance_error'] = (df['P_T'] - df['calculated_P']).abs()
    
    # 4. SYSTEM HEALTH INDEX (0-100)
    # Ref: IEC 61000-4-30
    # Higher error reduces the health score
    max_error = df['energy_imbalance_error'].max() if df['energy_imbalance_error'].max() > 0 else 1
    df['health_score'] = (1 - (df['energy_imbalance_error'] / max_error)) * 100
    df['health_score'] = df['health_score'].round(2)

    # 5. IMBALANCE LOSS INDICATOR
    # Quantifying current unbalance percentage
    df['i_unbalance_pct'] = ((df[['I_A', 'I_B', 'I_C']].max(axis=1) - i_avg) / i_avg.replace(0, 1)) * 100
    df['imbalance_loss_flag'] = df['i_unbalance_pct'].apply(lambda x: 1 if x > 15 else 0)

    # 6. Export Results
    df.to_excel(output_file, index=False)
    print(f"System Health Monitoring Dataset Created: {output_file}")

if __name__ == "__main__":
    generate_health_monitoring_dataset()