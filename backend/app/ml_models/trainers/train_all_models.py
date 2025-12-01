"""
Train All ML Models
Trains all 16 models using generated synthetic data
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow
try:
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. LSTM models will not be trained.")


class ModelTrainer:
    """Trains all 16 ML models for grid monitoring"""

    def __init__(self, data_path: str):
        """
        Initialize trainer with training data

        Args:
            data_path: Path to training_data.csv
        """
        print(f"Loading training data from {data_path}...")
        self.df = pd.read_csv(data_path)
        print(f"Loaded {len(self.df)} samples")

        self.models_dir = Path(__file__).parent.parent / "trained"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Prepare features
        self._prepare_features()

    def _prepare_features(self):
        """Extract and engineer features from raw data"""
        print("\nEngineering features...")

        # Calculate derived features
        self.df['v_avg'] = (self.df['v_a'] + self.df['v_b'] + self.df['v_c']) / 3
        self.df['i_avg'] = (self.df['i_a'] + self.df['i_b'] + self.df['i_c']) / 3
        self.df['p_total'] = self.df['p_a'] + self.df['p_b'] + self.df['p_c']
        self.df['q_total'] = self.df['q_a'] + self.df['q_b'] + self.df['q_c']

        # Voltage features
        self.df['v_variance'] = self.df[['v_a', 'v_b', 'v_c']].var(axis=1)
        self.df['v_imbalance'] = self.df.apply(
            lambda row: self._calc_imbalance(row['v_a'], row['v_b'], row['v_c']), axis=1
        )
        self.df['v_deviation'] = abs(self.df['v_avg'] - 230) / 230 * 100

        # Current features
        self.df['i_variance'] = self.df[['i_a', 'i_b', 'i_c']].var(axis=1)
        self.df['i_imbalance'] = self.df.apply(
            lambda row: self._calc_imbalance(row['i_a'], row['i_b'], row['i_c']), axis=1
        )

        # Power features
        self.df['s_total'] = np.sqrt(self.df['p_total']**2 + self.df['q_total']**2)
        self.df['power_factor'] = self.df['p_total'] / (self.df['s_total'] + 0.0001)
        self.df['power_factor'] = self.df['power_factor'].clip(0, 1)

        self.df['p_imbalance'] = self.df.apply(
            lambda row: self._calc_imbalance(row['p_a'], row['p_b'], row['p_c']), axis=1
        )

        # Frequency features
        self.df['f_deviation'] = abs(self.df['freq'] - 50.0)

        print("Features engineered successfully")

    def _calc_imbalance(self, a, b, c):
        """Calculate 3-phase imbalance percentage"""
        avg = (a + b + c) / 3
        if avg == 0:
            return 0
        max_dev = max(abs(a - avg), abs(b - avg), abs(c - avg))
        return (max_dev / avg) * 100

    # ==================== REAL-TIME MONITORING MODELS ====================

    def train_voltage_anomaly_detector(self):
        """Train Isolation Forest for voltage anomaly detection"""
        print("\n[1/16] Training Voltage Anomaly Detector (Isolation Forest)...")

        # Features MUST match model_manager.py exactly!
        # Lines 109-114 of model_manager.py use these 5 features:
        features = ['v_avg', 'v_variance', 'v_imbalance_pct', 'v_deviation_pct', 'v_rate_of_change']
        X = self.df[features].fillna(0)

        # Isolation Forest (unsupervised)
        model = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        model.fit(X)

        # Save model
        joblib.dump(model, self.models_dir / 'voltage_anomaly_detector.pkl')
        print("✓ Voltage Anomaly Detector trained and saved")

    def train_harmonic_analyzer(self):
        """Train Random Forest for harmonic analysis"""
        print("\n[2/16] Training Harmonic Analyzer (Random Forest)...")

        # Create labels based on voltage variance (proxy for THD)
        self.df['thd_level'] = pd.cut(
            self.df['v_variance'],
            bins=[0, 10, 25, 100],
            labels=['Low', 'Medium', 'High']
        )

        # Features MUST match model_manager.py analyze_harmonics() exactly!
        features = ['v_thd_estimated', 'v_variance', 'power_factor', 'v_quality_score']
        X = self.df[features].fillna(0)
        y = self.df['thd_level'].fillna('Low')

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        model.fit(X, y)

        joblib.dump(model, self.models_dir / 'harmonic_analyzer.pkl')
        print("✓ Harmonic Analyzer trained and saved")

    def train_frequency_stability_predictor(self):
        """Train LSTM for frequency stability prediction"""
        print("\n[3/16] Training Frequency Stability Predictor (LSTM)...")

        if not TENSORFLOW_AVAILABLE:
            print("⚠ TensorFlow not available, skipping LSTM training")
            return

        # Prepare sequence data
        timeseries_data = self.df[self.df['scenario'] == 'timeseries'].copy()

        if len(timeseries_data) < 200:
            print("⚠ Not enough time-series data, using mock model")
            return

        # Create sequences
        freq_values = timeseries_data['freq'].values
        X_seq, y_seq = [], []

        seq_length = 100
        for i in range(len(freq_values) - seq_length - 10):
            X_seq.append(freq_values[i:i+seq_length])
            y_seq.append(freq_values[i+seq_length:i+seq_length+10])

        X_seq = np.array(X_seq).reshape(-1, seq_length, 1)
        y_seq = np.array(y_seq)

        # Build LSTM model
        model = keras.Sequential([
            layers.LSTM(64, input_shape=(seq_length, 1), return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(10)
        ])

        model.compile(optimizer='adam', loss='mse')
        model.fit(X_seq, y_seq, epochs=10, batch_size=32, verbose=0)

        model.save(self.models_dir / 'frequency_stability_predictor.h5')
        print("✓ Frequency Stability Predictor trained and saved")

    def train_phase_imbalance_classifier(self):
        """Train Decision Tree for phase imbalance classification"""
        print("\n[4/16] Training Phase Imbalance Classifier (Decision Tree)...")

        # Create severity labels
        def get_severity(row):
            max_imb = max(row['v_imbalance'], row['i_imbalance'], row['p_imbalance'])
            if max_imb > 15:
                return 'Critical'
            elif max_imb > 8:
                return 'Warning'
            return 'Normal'

        self.df['imbalance_severity'] = self.df.apply(get_severity, axis=1)

        # Features MUST match model_manager.py classify_phase_imbalance() exactly!
        features = ['v_imbalance', 'i_imbalance', 'p_imbalance', 'overall_balance_score']
        X = self.df[features].fillna(0)
        y = self.df['imbalance_severity']

        model = DecisionTreeClassifier(
            max_depth=8,
            random_state=42,
            min_samples_split=20
        )
        model.fit(X, y)

        joblib.dump(model, self.models_dir / 'phase_imbalance_classifier.pkl')
        print("✓ Phase Imbalance Classifier trained and saved")

    # ==================== PREDICTIVE MAINTENANCE MODELS ====================

    def train_equipment_failure_predictor(self):
        """Train XGBoost for equipment failure prediction"""
        print("\n[5/16] Training Equipment Failure Predictor (XGBoost)...")

        # Create failure labels (overload + poor conditions = failure risk)
        def get_failure_label(row):
            risk_score = 0
            if row['i_avg'] > 75:  # High current
                risk_score += 0.3
            if row['v_variance'] > 20:
                risk_score += 0.2
            if row['i_variance'] > 20:
                risk_score += 0.2
            if row['power_factor'] < 0.85:
                risk_score += 0.2
            if row['v_imbalance'] > 10:
                risk_score += 0.1
            return 1 if risk_score > 0.5 else 0

        self.df['failure_risk'] = self.df.apply(get_failure_label, axis=1)

        # Features MUST match model_manager.py lines 239-245 exactly!
        features = ['i_avg', 'i_variance', 'v_variance', 'power_factor', 'i_spike_detected', 'v_imbalance_pct']
        X = self.df[features].fillna(0)
        y = self.df['failure_risk']

        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X, y)

        joblib.dump(model, self.models_dir / 'equipment_failure_predictor.pkl')
        print("✓ Equipment Failure Predictor trained and saved")

    def train_overload_risk_classifier(self):
        """Train SVM for overload risk classification"""
        print("\n[6/16] Training Overload Risk Classifier (SVM)...")

        # Create risk labels
        def get_overload_risk(i_avg):
            if i_avg > 85:
                return 'High'
            elif i_avg > 65:
                return 'Medium'
            return 'Low'

        self.df['overload_risk'] = self.df['i_avg'].apply(get_overload_risk)

        # Features MUST match model_manager.py classify_overload_risk() exactly!
        features = ['i_avg', 'p_total', 'i_max_phase', 'i_imbalance_pct']
        X = self.df[features].fillna(0)
        y = self.df['overload_risk']

        # Standardize features for SVM
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = SVC(kernel='rbf', probability=True, random_state=42)
        model.fit(X_scaled, y)

        # Save both model and scaler
        joblib.dump({'model': model, 'scaler': scaler},
                   self.models_dir / 'overload_risk_classifier.pkl')
        print("✓ Overload Risk Classifier trained and saved")

    def train_power_quality_index_model(self):
        """Train Neural Network for Power Quality Index"""
        print("\n[7/16] Training Power Quality Index Model (Neural Network)...")

        if not TENSORFLOW_AVAILABLE:
            print("⚠ TensorFlow not available, using Random Forest instead")
            self._train_pqi_alternative()
            return

        # Calculate PQI score
        def calc_pqi(row):
            v_quality = max(0, 100 - row['v_deviation'] * 10)
            f_quality = max(0, 100 - row['f_deviation'] * 200)
            pf_quality = row['power_factor'] * 100
            return (v_quality * 0.4 + f_quality * 0.3 + pf_quality * 0.3)

        self.df['pqi'] = self.df.apply(calc_pqi, axis=1)

        # Features MUST match model_manager.py calculate_power_quality_index() exactly!
        features = ['v_quality_score', 'f_quality_score', 'pf_quality_score', 'v_thd_estimated', 'overall_balance_score']
        X = self.df[features].fillna(0).values
        y = self.df['pqi'].values

        # Build neural network
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(5,)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='linear')
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.fit(X, y, epochs=20, batch_size=32, verbose=0)

        model.save(self.models_dir / 'power_quality_index.h5')
        print("✓ Power Quality Index Model trained and saved")

    def _train_pqi_alternative(self):
        """Alternative PQI model using Random Forest"""
        def calc_pqi(row):
            v_quality = max(0, 100 - row['v_deviation'] * 10)
            f_quality = max(0, 100 - row['f_deviation'] * 200)
            pf_quality = row['power_factor'] * 100
            return (v_quality * 0.4 + f_quality * 0.3 + pf_quality * 0.3)

        self.df['pqi'] = self.df.apply(calc_pqi, axis=1)

        # Features MUST match model_manager.py calculate_power_quality_index() exactly!
        features = ['v_quality_score', 'f_quality_score', 'pf_quality_score', 'v_thd_estimated', 'overall_balance_score']
        X = self.df[features].fillna(0)
        y = self.df['pqi']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        joblib.dump(model, self.models_dir / 'power_quality_index.pkl')

    def train_voltage_sag_predictor(self):
        """Train Random Forest for voltage sag prediction"""
        print("\n[8/16] Training Voltage Sag Predictor (Random Forest)...")

        # Create sag labels
        self.df['voltage_sag'] = (self.df['v_avg'] < 207).astype(int)  # < 0.9 pu

        # Features MUST match model_manager.py predict_voltage_sag() exactly!
        features = ['v_avg', 'voltage_avg_std', 'v_rate_of_change', 'voltage_avg_trend']
        X = self.df[features].fillna(0)
        y = self.df['voltage_sag']

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X, y)

        joblib.dump(model, self.models_dir / 'voltage_sag_predictor.pkl')
        print("✓ Voltage Sag Predictor trained and saved")

    # ==================== ENERGY FLOW MODELS ====================

    def train_load_forecasting_model(self):
        """Train LSTM for load forecasting (simplified to Linear Regression)"""
        print("\n[9/16] Training Load Forecasting Model (Linear Regression)...")

        # Use rolling statistics as features
        if 'timestamp' in self.df.columns:
            ts_data = self.df.dropna(subset=['p_total']).copy()
            ts_data['hour'] = pd.to_datetime(ts_data['timestamp']).dt.hour
            ts_data['day_of_week'] = pd.to_datetime(ts_data['timestamp']).dt.dayofweek

            features = ['hour', 'day_of_week', 'i_avg']
            X = ts_data[features].fillna(0)
            y = ts_data['p_total']
        else:
            features = ['i_avg', 'v_avg', 'power_factor']
            X = self.df[features].fillna(0)
            y = self.df['p_total']

        model = LinearRegression()
        model.fit(X, y)

        joblib.dump(model, self.models_dir / 'load_forecasting_lstm.pkl')
        print("✓ Load Forecasting Model trained and saved")

    def train_energy_loss_estimator(self):
        """Train Linear Regression for energy loss estimation"""
        print("\n[10/16] Training Energy Loss Estimator (Linear Regression)...")

        # Estimate losses based on I²R and imbalance
        self.df['estimated_loss'] = (
            self.df['i_avg']**2 * 0.1 +  # Resistive losses
            self.df['i_imbalance'] * 0.05  # Imbalance losses
        )

        # Features MUST match model_manager.py estimate_energy_loss() exactly!
        features = ['i_avg', 'p_total', 'i_imbalance_pct', 'power_factor']
        X = self.df[features].fillna(0)
        y = self.df['estimated_loss']

        model = LinearRegression()
        model.fit(X, y)

        joblib.dump(model, self.models_dir / 'energy_loss_estimator.pkl')
        print("✓ Energy Loss Estimator trained and saved")

    def train_power_flow_optimizer(self):
        """Train optimization model for power flow (simplified)"""
        print("\n[11/16] Training Power Flow Optimizer (Mock Model)...")

        # This would typically use scipy.optimize or OR-tools
        # For now, save a simple rule-based model
        model = {'type': 'rule_based', 'version': '1.0'}

        joblib.dump(model, self.models_dir / 'power_flow_optimizer.pkl')
        print("✓ Power Flow Optimizer saved")

    def train_demand_response_model(self):
        """Train K-Means for demand response clustering"""
        print("\n[12/16] Training Demand Response Model (K-Means)...")

        # Features MUST match model_manager.py assess_demand_response() exactly!
        features = ['p_total', 'active_power_mean', 'active_power_std']
        X = self.df[features].fillna(0)

        model = KMeans(n_clusters=3, random_state=42, n_init=10)
        model.fit(X)

        joblib.dump(model, self.models_dir / 'demand_response_potential.pkl')
        print("✓ Demand Response Model trained and saved")

    # ==================== DECISION MAKING MODELS ====================

    def train_reactive_compensation_model(self):
        """Train Neural Network for reactive power compensation"""
        print("\n[13/16] Training Reactive Compensation Model (Linear Regression)...")

        # Calculate optimal reactive power
        target_pf = 0.95
        self.df['optimal_q'] = self.df.apply(
            lambda row: row['p_total'] * (
                np.tan(np.arccos(row['power_factor'])) -
                np.tan(np.arccos(target_pf))
            ) if row['power_factor'] < target_pf else 0,
            axis=1
        )

        features = ['power_factor', 'q_total', 'p_total']
        X = self.df[features].fillna(0)
        y = self.df['optimal_q']

        model = LinearRegression()
        model.fit(X, y)

        joblib.dump(model, self.models_dir / 'reactive_power_compensator.pkl')
        print("✓ Reactive Compensation Model trained and saved")

    def train_load_balancing_optimizer(self):
        """Train load balancing optimizer (rule-based)"""
        print("\n[14/16] Training Load Balancing Optimizer (Mock Model)...")

        model = {'type': 'mcda', 'version': '1.0'}
        joblib.dump(model, self.models_dir / 'load_balancing_optimizer.pkl')
        print("✓ Load Balancing Optimizer saved")

    def train_grid_stability_scorer(self):
        """Train ensemble model for grid stability scoring"""
        print("\n[15/16] Training Grid Stability Scorer (Random Forest)...")

        # Calculate stability score
        def calc_stability(row):
            v_score = 100 - abs(row['v_avg'] - 230) / 230 * 100
            f_score = 100 - abs(row['freq'] - 50) * 100
            pf_score = row['power_factor'] * 100
            balance_score = 100 - max(row['v_imbalance'], row['i_imbalance'])
            return (v_score + f_score + pf_score + balance_score) / 400

        self.df['stability_score'] = self.df.apply(calc_stability, axis=1)

        # Features MUST match model_manager.py score_grid_stability() exactly!
        features = ['v_avg', 'f_value', 'power_factor', 'overall_balance_score', 'power_quality_index']
        X = self.df[features].fillna(0)
        y = self.df['stability_score']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        joblib.dump(model, self.models_dir / 'grid_stability_scorer.pkl')
        print("✓ Grid Stability Scorer trained and saved")

    def train_optimal_dispatch_advisor(self):
        """Train SVR for optimal generation dispatch"""
        print("\n[16/16] Training Optimal Dispatch Advisor (SVR)...")

        # Optimal generation = current load + 15% reserve
        self.df['optimal_generation'] = self.df['p_total'] * 1.15

        # Features MUST match model_manager.py advise_optimal_dispatch() exactly!
        features = ['p_total', 'active_power_mean', 'active_power_trend']
        X = self.df[features].fillna(0)
        y = self.df['optimal_generation']

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = SVR(kernel='rbf')
        model.fit(X_scaled, y)

        joblib.dump({'model': model, 'scaler': scaler},
                   self.models_dir / 'optimal_dispatch_advisor.pkl')
        print("✓ Optimal Dispatch Advisor trained and saved")

    def train_all(self):
        """Train all 16 models sequentially"""
        print("\n" + "="*60)
        print("TRAINING ALL ML MODELS")
        print("="*60)

        self.train_voltage_anomaly_detector()
        self.train_harmonic_analyzer()
        self.train_frequency_stability_predictor()
        self.train_phase_imbalance_classifier()

        self.train_equipment_failure_predictor()
        self.train_overload_risk_classifier()
        self.train_power_quality_index_model()
        self.train_voltage_sag_predictor()

        self.train_load_forecasting_model()
        self.train_energy_loss_estimator()
        self.train_power_flow_optimizer()
        self.train_demand_response_model()

        self.train_reactive_compensation_model()
        self.train_load_balancing_optimizer()
        self.train_grid_stability_scorer()
        self.train_optimal_dispatch_advisor()

        print("\n" + "="*60)
        print("✓ ALL MODELS TRAINED SUCCESSFULLY!")
        print("="*60)


if __name__ == "__main__":
    # Path to training data
    data_path = Path(__file__).parent.parent / "trained" / "training_data.csv"

    if not data_path.exists():
        print(f"Error: Training data not found at {data_path}")
        print("Please run data_generator.py first to generate training data.")
        exit(1)

    # Train all models
    trainer = ModelTrainer(str(data_path))
    trainer.train_all()
