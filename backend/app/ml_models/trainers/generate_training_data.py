#!/usr/bin/env python3
"""
Generate Synthetic Training Data for ML Models
Creates realistic 3-phase power grid measurements
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_training_data(num_samples=5000):
    """
    Generate synthetic training data with realistic grid measurements

    Args:
        num_samples: Number of samples to generate (default: 5000)

    Returns:
        DataFrame with raw measurements
    """
    print(f"Generating {num_samples} training samples...")

    np.random.seed(42)

    data = []
    for i in range(num_samples):
        # Nominal values
        v_nominal = 230  # V
        i_nominal = 50   # A
        freq_nominal = 50.0  # Hz

        # Add realistic variations
        v_variation = np.random.normal(0, 5)
        i_variation = np.random.normal(0, 10)
        freq_variation = np.random.normal(0, 0.1)

        # Imbalance factor (0 = perfect balance, up to 0.2 = 20% imbalance)
        imbalance_factor = np.random.uniform(0, 0.2)

        # Phase A (reference)
        v_a = v_nominal + v_variation
        i_a = i_nominal + i_variation

        # Phase B (with imbalance)
        v_b = v_nominal + v_variation + np.random.normal(0, v_nominal * imbalance_factor)
        i_b = i_nominal + i_variation + np.random.normal(0, i_nominal * imbalance_factor)

        # Phase C (with imbalance)
        v_c = v_nominal + v_variation + np.random.normal(0, v_nominal * imbalance_factor)
        i_c = i_nominal + i_variation + np.random.normal(0, i_nominal * imbalance_factor)

        # Power factor (0.7 to 1.0)
        pf = np.random.uniform(0.7, 1.0)

        # Active and reactive power per phase
        p_a = v_a * i_a * pf / 1000  # kW
        p_b = v_b * i_b * pf / 1000
        p_c = v_c * i_c * pf / 1000

        q_a = v_a * i_a * np.sqrt(1 - pf**2) / 1000  # kVAR
        q_b = v_b * i_b * np.sqrt(1 - pf**2) / 1000
        q_c = v_c * i_c * np.sqrt(1 - pf**2) / 1000

        # Frequency
        freq = freq_nominal + freq_variation

        data.append({
            'v_a': v_a,
            'v_b': v_b,
            'v_c': v_c,
            'i_a': i_a,
            'i_b': i_b,
            'i_c': i_c,
            'p_a': p_a,
            'p_b': p_b,
            'p_c': p_c,
            'q_a': q_a,
            'q_b': q_b,
            'q_c': q_c,
            'freq': freq
        })

    df = pd.DataFrame(data)
    print(f"✓ Generated {len(df)} samples with {len(df.columns)} raw features")
    return df


if __name__ == "__main__":
    # Output path
    output_path = Path(__file__).parent.parent / "trained" / "training_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate data
    df = generate_training_data(num_samples=5000)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\n✓ Training data saved to: {output_path}")
    print(f"  Samples: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\nNext step: Run 'python train_all_models.py' to train all ML models")
