#!/usr/bin/env python3
"""
Setup script for ML models
Generates training data and trains all 16 models
"""

import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.ml_models.trainers.data_generator import GridDataGenerator
from app.ml_models.trainers.train_all_models import ModelTrainer


def main():
    print("="*70)
    print(" REMHART DIGITAL TWIN - ML MODEL SETUP")
    print("="*70)

    # Step 1: Generate training data
    print("\n[STEP 1/2] Generating training data...")
    print("-" * 70)

    generator = GridDataGenerator(seed=42)
    dataset = generator.generate_comprehensive_dataset()

    output_dir = Path(__file__).parent / "app" / "ml_models" / "trained"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "training_data.csv"

    generator.save_dataset(dataset, str(data_path))

    # Step 2: Train all models
    print("\n[STEP 2/2] Training all ML models...")
    print("-" * 70)

    trainer = ModelTrainer(str(data_path))
    trainer.train_all()

    print("\n" + "="*70)
    print(" ✓ ML SETUP COMPLETE!")
    print("="*70)
    print("\nAll 16 models are now trained and ready for inference.")
    print("You can now start the backend server and use the ML features.")


if __name__ == "__main__":
    main()
