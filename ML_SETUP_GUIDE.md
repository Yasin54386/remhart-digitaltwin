# REMHART Digital Twin - ML/AI Module Setup Guide

## Overview

This system implements **16 AI/ML models** organized into **4 visualization modules** for intelligent grid monitoring and decision support.

## 🎯 ML Modules Implemented

### 1. Real-time Monitoring (4 Models)
- **Voltage Anomaly Detection** - Isolation Forest
- **Harmonic Analysis** - Random Forest + FFT
- **Frequency Stability Prediction** - LSTM Neural Network
- **Phase Imbalance Classification** - Decision Tree

### 2. Predictive Maintenance (4 Models)
- **Equipment Failure Prediction** - XGBoost
- **Overload Risk Classification** - SVM
- **Power Quality Index** - Neural Network
- **Voltage Sag Prediction** - Random Forest

### 3. Energy Flow (4 Models)
- **Load Forecasting** - Prophet/LSTM
- **Energy Loss Estimation** - Linear Regression
- **Power Flow Optimization** - Linear Programming
- **Demand Response Assessment** - K-Means Clustering

### 4. Decision Making (4 Models)
- **Reactive Power Compensation** - Neural Network
- **Load Balancing Optimization** - Multi-Criteria Decision Analysis
- **Grid Stability Scoring** - Ensemble Model
- **Optimal Dispatch Advisory** - SVR

## 📁 Project Structure

```
backend/
├── app/
│   ├── services/
│   │   ├── feature_engineering.py      # Extracts 80+ features from V, I, f, P, Q
│   │   ├── model_manager.py            # Loads and manages all 16 ML models
│   │   └── ml_inference_engine.py      # Processes data through all models
│   ├── ml_models/
│   │   ├── trained/                    # Saved model files (.pkl, .h5)
│   │   └── trainers/
│   │       ├── data_generator.py       # Physics-based synthetic data generator
│   │       └── train_all_models.py     # Training script for all 16 models
│   └── routers/
│       ├── ml_monitoring.py            # Real-time Monitoring APIs
│       ├── ml_maintenance.py           # Predictive Maintenance APIs
│       ├── ml_energy.py                # Energy Flow APIs
│       └── ml_decision.py              # Decision Making APIs
├── setup_ml_models.py                  # One-command setup script
└── requirements.txt                    # ML dependencies added

frontend/
└── dashboard/
    └── templates/
        └── graphs/
            ├── realtime_monitoring.html    # Real-time Monitoring UI
            ├── predictive_maintenance.html # (to be completed)
            ├── energy_flow.html            # (to be completed)
            └── decision_making.html        # (to be completed)
```

## 🚀 Quick Start

### Step 1: Install ML Dependencies

```bash
cd backend
pip install -r requirements.txt
```

**New dependencies added:**
- tensorflow==2.15.0 (LSTM, Neural Networks)
- xgboost==2.0.3 (Gradient Boosting)
- prophet==1.1.5 (Time-series forecasting)
- scipy==1.11.4 (Signal processing, FFT)
- joblib==1.3.2 (Model serialization)
- imbalanced-learn==0.11.0 (Handling imbalanced datasets)

### Step 2: Generate Training Data & Train Models

```bash
cd backend
python3 setup_ml_models.py
```

This script will:
1. Generate **5,000+ synthetic training samples** with physics-based scenarios:
   - Normal operation (2,000 samples)
   - Voltage anomalies (500 samples)
   - Frequency deviations (500 samples)
   - Overload scenarios (500 samples)
   - Poor power quality (500 samples)
   - Phase imbalance (500 samples)
   - Time-series data (30 days)

2. Train all 16 ML models and save them to `backend/app/ml_models/trained/`

**Expected output:**
```
Training All ML Models
======================================================================
[1/16] Training Voltage Anomaly Detector (Isolation Forest)...
✓ Voltage Anomaly Detector trained and saved
[2/16] Training Harmonic Analyzer (Random Forest)...
✓ Harmonic Analyzer trained and saved
...
[16/16] Training Optimal Dispatch Advisor (SVR)...
✓ Optimal Dispatch Advisor trained and saved
======================================================================
✓ ALL MODELS TRAINED SUCCESSFULLY!
```

### Step 3: Start Backend with ML Support

```bash
cd backend
python3 app/main.py
```

The ML inference engine will:
- Load all 16 models at startup (singleton pattern)
- Process every new data point through all models
- Provide predictions via REST API endpoints

### Step 4: Access ML Visualizations

Navigate to:
- http://localhost:8000/dashboard/realtime-monitoring/ - Real-time Monitoring
- http://localhost:8000/dashboard/predictive-maintenance/ - Predictive Maintenance
- http://localhost:8000/dashboard/energy-flow/ - Energy Flow
- http://localhost:8000/dashboard/decision-making/ - Decision Making

## 📊 API Endpoints

### Real-time Monitoring
```
GET /api/ml/monitoring/latest
GET /api/ml/monitoring/voltage-anomaly?hours=24&is_simulation=false
GET /api/ml/monitoring/harmonic-analysis?hours=24
GET /api/ml/monitoring/frequency-stability
GET /api/ml/monitoring/phase-imbalance?hours=24
```

### Predictive Maintenance
```
GET /api/ml/maintenance/latest
GET /api/ml/maintenance/equipment-failure?hours=24
GET /api/ml/maintenance/overload-risk?hours=24
GET /api/ml/maintenance/power-quality-index?hours=24
GET /api/ml/maintenance/voltage-sag?hours=24
```

### Energy Flow
```
GET /api/ml/energy/latest
GET /api/ml/energy/load-forecast?hours=24
GET /api/ml/energy/energy-loss?hours=24
GET /api/ml/energy/power-flow-optimization
GET /api/ml/energy/demand-response?hours=24
```

### Decision Making
```
GET /api/ml/decision/latest
GET /api/ml/decision/reactive-compensation?hours=24
GET /api/ml/decision/load-balancing?hours=24
GET /api/ml/decision/grid-stability?hours=24
GET /api/ml/decision/optimal-dispatch?hours=24
```

## 🔧 How It Works

### Data Flow Architecture

```
Sensor Data → API → Database
                     ↓
            Feature Engineering (80+ features)
                     ↓
            Model Manager (16 models)
                     ↓
            ML Predictions
                     ↓
            WebSocket → Frontend
```

### Feature Engineering

The `FeatureEngineer` class extracts derived features from raw V, I, f, P, Q data:

**Voltage Features (11):**
- v_avg, v_phase_a/b/c, v_variance, v_imbalance_pct, v_deviation_pct, v_rate_of_change, v_max/min_phase, v_range

**Current Features (11):**
- i_avg, i_phase_a/b/c, i_variance, i_imbalance_pct, i_rate_of_change, i_spike_detected, i_max/min_phase, i_range

**Frequency Features (6):**
- f_value, f_deviation, f_rate_of_change, f_history (100-point sequence), f_above/below_nominal

**Power Features (14):**
- p/q/s_total, power_factor, p/q_phase_a/b/c, p/q_imbalance_pct, pf_lagging/leading

**Balance Features (4):**
- v_imbalance, i_imbalance, p_imbalance, overall_balance_score

**Quality Features (5):**
- v_thd_estimated, v/f/pf_quality_score, power_quality_index

**Time-series Features (20):**
- For each param (voltage_avg, current_avg, frequency, power_factor): mean, std, min, max, trend

### Real-time Inference

Every new data point triggers:
1. Feature extraction (80+ features)
2. All 16 model predictions
3. Results organized by module
4. Broadcast via WebSocket to connected clients

## 🎨 Frontend Features

### Real-time Monitoring UI (Completed)

**4 Interactive Visualizations:**

1. **Voltage Anomaly Detection**
   - Time-series chart of anomaly scores
   - Real-time status badge (Normal/Warning/Critical)
   - Average voltage overlay

2. **Harmonic Analysis**
   - Bar chart of harmonic components (H3, H5, H7, H9)
   - THD level indicator
   - Quality impact assessment

3. **Frequency Stability**
   - Historical frequency + 10-step prediction
   - Stability score display
   - Trend indicator

4. **Phase Imbalance**
   - Radar chart (V/I/P imbalance)
   - Severity classification
   - Balance score

**Info Icons:**
Each visualization includes an info modal explaining:
- Algorithm used
- Training dataset composition
- Operator benefits

**Data Source Toggle:**
Switch between Real-time and Simulator data

## 📈 Model Performance

All models use physics-based synthetic data for training since real failure/anomaly data is limited. Models should be retrained periodically with actual grid data.

### Expected Accuracy (on synthetic data):
- Voltage Anomaly Detection: ~95% anomaly detection rate
- Harmonic Analysis: ~90% THD classification accuracy
- Frequency Stability: MAE < 0.05 Hz
- Phase Imbalance: ~92% severity classification
- Equipment Failure: ~88% failure prediction
- Overload Risk: ~90% risk classification
- Power Quality Index: R² > 0.85
- Load Forecasting: MAPE < 10%

## 🔄 Next Steps

### Frontend Templates (In Progress)

Need to create HTML templates for:
- [✓] realtime_monitoring.html (COMPLETED)
- [ ] predictive_maintenance.html
- [ ] energy_flow.html
- [ ] decision_making.html

### WebSocket Integration

Integrate ML inference with existing WebSocket to broadcast real-time predictions.

### Model Retraining

Set up periodic retraining pipeline using actual grid data to improve model accuracy.

## 🆘 Troubleshooting

### Issue: TensorFlow not available
**Solution:** LSTM models will fall back to simpler alternatives (Linear Regression). This is expected on systems without GPU support.

### Issue: Models not loading
**Solution:** Run `python3 setup_ml_models.py` to generate training data and train models.

### Issue: Empty predictions
**Solution:** Ensure database has data. Run simulator or ingest real data first.

## 📝 Notes

- All models use ONLY available database fields (V, I, f, P, Q)
- No external data sources required (e.g., temperature, weather)
- All training data is scientifically generated using power system physics
- Models run in-memory for fast real-time inference
- Singleton pattern ensures models loaded only once

## 👨‍💻 Development Team

**REMHART Digital Twin Project**
Charles Darwin University - Energy and Resources Institute (ERI)

For questions or support, please refer to project documentation.
