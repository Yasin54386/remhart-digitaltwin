# REMHART Digital Twin - 3-Tier System Architecture

## Overview

The REMHART Smart Grid Digital Twin employs a **modern 3-tier microservices architecture** with clear separation of concerns, enabling scalability, maintainability, and independent deployment of components.

---

## Architecture Diagram

```
┌───────────────────────────────────────────────────────────────────────────┐
│                            TIER 1: PRESENTATION LAYER                      │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                         Django Frontend                               │ │
│  │                         Version: 5.2.8                                │ │
│  │                         Port: 8000                                    │ │
│  │                         Protocol: HTTP/HTTPS                          │ │
│  │                                                                        │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │ │
│  │  │  Landing    │  │ Login/Auth  │  │  Dashboard   │  │ Simulator  │ │ │
│  │  │  Page       │  │  Views      │  │  Views       │  │  UI        │ │ │
│  │  └─────────────┘  └─────────────┘  └──────────────┘  └────────────┘ │ │
│  │                                                                        │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │              Real-time Visualization Components                  │ │ │
│  │  ├──────────────┬──────────────────┬─────────────┬─────────────────┤ │ │
│  │  │ Real-time    │ Predictive       │ Energy Flow │ Decision Making │ │ │
│  │  │ Monitoring   │ Maintenance      │ Analytics   │ Dashboard       │ │ │
│  │  └──────────────┴──────────────────┴─────────────┴─────────────────┘ │ │
│  │                                                                        │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │                    Static Assets                                 │ │ │
│  │  │    - Chart.js / D3.js (Visualizations)                           │ │ │
│  │  │    - Bootstrap / Tailwind CSS (Styling)                          │ │ │
│  │  │    - WebSocket Client (Real-time updates)                        │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                        │ │
│  │  Database: SQLite3 (db.sqlite3) - Session storage only               │ │
│  └────────────────────────────────┬───────────────────────────────────────┘ │
│                                   │ HTTP/REST API + WebSocket              │
└───────────────────────────────────┼───────────────────────────────────────┘
                                    │
                                    │
┌───────────────────────────────────▼───────────────────────────────────────┐
│                     TIER 2: APPLICATION / BUSINESS LOGIC LAYER             │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                        FastAPI Backend                                │ │
│  │                        Version: 0.104.1                               │ │
│  │                        Port: 8001                                     │ │
│  │                        Protocol: HTTP/HTTPS + WebSocket               │ │
│  │                                                                        │ │
│  │  ┌────────────────────────────────────────────────────────────────┐  │ │
│  │  │                       API Routers                               │  │ │
│  │  ├────────────┬───────────────┬────────────┬─────────────────────┤  │ │
│  │  │ /api/auth  │ /api/grid     │ /api/sim   │ /api/ml/*           │  │ │
│  │  │ (Login)    │ (Grid Data)   │ (Simulator)│ (ML Endpoints)      │  │ │
│  │  │            │               │            │ - monitoring        │  │ │
│  │  │            │               │            │ - maintenance       │  │ │
│  │  │            │               │            │ - energy            │  │ │
│  │  │            │               │            │ - decision          │  │ │
│  │  └────────────┴───────────────┴────────────┴─────────────────────┘  │ │
│  │                                                                        │ │
│  │  ┌────────────────────────────────────────────────────────────────┐  │ │
│  │  │                  WebSocket Server                               │  │ │
│  │  │                  Endpoint: /ws/grid-data                        │  │ │
│  │  │                  Real-time bi-directional communication         │  │ │
│  │  └────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                        │ │
│  │  ┌────────────────────────────────────────────────────────────────┐  │ │
│  │  │                     Business Logic Services                     │  │ │
│  │  ├──────────────────────────┬──────────────────────────────────────┤  │ │
│  │  │ Feature Engineering      │  ML Inference Engine                │  │ │
│  │  │ - 80+ features           │  - 16 models across 4 modules       │  │ │
│  │  │ - Voltage, Current       │  - Singleton ModelManager           │  │ │
│  │  │ - Power, Frequency       │  - <100ms inference latency         │  │ │
│  │  │ - Time-series stats      │                                      │  │ │
│  │  ├──────────────────────────┼──────────────────────────────────────┤  │ │
│  │  │ Data Generator           │  Simulation Engine                  │  │ │
│  │  │ - Test data synthesis    │  - 5 scenario types                 │  │ │
│  │  │ - Realistic patterns     │  - Configurable parameters          │  │ │
│  │  └──────────────────────────┴──────────────────────────────────────┘  │ │
│  │                                                                        │ │
│  │  ┌────────────────────────────────────────────────────────────────┐  │ │
│  │  │                   Validation & Schemas                          │  │ │
│  │  │                   (Pydantic Models)                             │  │ │
│  │  │   - GridDataCreate, GridDataResponse                           │  │ │
│  │  │   - SimulationCreate, SimulationResponse                       │  │ │
│  │  │   - MLPredictionResponse                                       │  │ │
│  │  └────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                        │ │
│  │  ┌────────────────────────────────────────────────────────────────┐  │ │
│  │  │                   Security & Middleware                         │  │ │
│  │  │   - JWT Authentication (Bearer tokens)                         │  │ │
│  │  │   - RBAC (admin, operator, analyst, viewer)                    │  │ │
│  │  │   - CORS (localhost:8000, 8001, 3000)                          │  │ │
│  │  │   - Request logging & error handling                           │  │ │
│  │  └────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                        │ │
│  │  ML Models: 16 trained models (.pkl, .h5 files)                      │ │
│  └────────────────────────────────┬───────────────────────────────────────┘ │
│                                   │ SQLAlchemy ORM + MySQL Connector       │
└───────────────────────────────────┼───────────────────────────────────────┘
                                    │
                                    │
┌───────────────────────────────────▼───────────────────────────────────────┐
│                         TIER 3: DATA / PERSISTENCE LAYER                   │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                         MySQL Database                                │ │
│  │                         Database: remhart_db                          │ │
│  │                         ORM: SQLAlchemy 2.0.23                        │ │
│  │                         Migration: Alembic                            │ │
│  │                                                                        │ │
│  │  ┌────────────────────────────────────────────────────────────────┐  │ │
│  │  │                      Database Tables                            │  │ │
│  │  ├────────────────────────┬───────────────────────────────────────┤  │ │
│  │  │ DateTimeTable          │  Core timestamp reference            │  │ │
│  │  │ - id (PK)              │  - Microsecond precision             │  │ │
│  │  │ - timestamp            │  - Simulation metadata               │  │ │
│  │  │ - is_simulation        │                                       │  │ │
│  │  │ - simulation_id        │                                       │  │ │
│  │  ├────────────────────────┼───────────────────────────────────────┤  │ │
│  │  │ VoltageTable           │  3-phase voltage measurements        │  │ │
│  │  │ - id (PK)              │  - phaseA, phaseB, phaseC            │  │ │
│  │  │ - timestamp_id (FK)    │  - average (calculated)              │  │ │
│  │  │                        │  - Range: 0-300V                      │  │ │
│  │  ├────────────────────────┼───────────────────────────────────────┤  │ │
│  │  │ CurrentTable           │  3-phase current measurements        │  │ │
│  │  │ - id (PK)              │  - phaseA, phaseB, phaseC            │  │ │
│  │  │ - timestamp_id (FK)    │  - average (calculated)              │  │ │
│  │  │                        │  - Unit: Amperes                      │  │ │
│  │  ├────────────────────────┼───────────────────────────────────────┤  │ │
│  │  │ FrequencyTable         │  Grid frequency                      │  │ │
│  │  │ - id (PK)              │  - frequency_value                   │  │ │
│  │  │ - timestamp_id (FK)    │  - Range: 49.0-51.0 Hz               │  │ │
│  │  │                        │  - Nominal: 50 Hz (Australia)        │  │ │
│  │  ├────────────────────────┼───────────────────────────────────────┤  │ │
│  │  │ ActivePowerTable       │  Real power (3-phase)                │  │ │
│  │  │ - id (PK)              │  - phaseA, phaseB, phaseC            │  │ │
│  │  │ - timestamp_id (FK)    │  - total (sum)                       │  │ │
│  │  │                        │  - Unit: Watts                        │  │ │
│  │  ├────────────────────────┼───────────────────────────────────────┤  │ │
│  │  │ ReactivePowerTable     │  Reactive power (3-phase)            │  │ │
│  │  │ - id (PK)              │  - phaseA, phaseB, phaseC            │  │ │
│  │  │ - timestamp_id (FK)    │  - total (sum)                       │  │ │
│  │  │                        │  - Unit: VAR                          │  │ │
│  │  ├────────────────────────┼───────────────────────────────────────┤  │ │
│  │  │ SimulationMetadata     │  Simulation tracking                 │  │ │
│  │  │ - id (PK)              │  - simulation_id (unique)            │  │ │
│  │  │ - name, scenario_type  │  - parameters (JSON)                 │  │ │
│  │  │ - is_active, status    │  - created_by, created_at            │  │ │
│  │  ├────────────────────────┼───────────────────────────────────────┤  │ │
│  │  │ User                   │  Authentication & RBAC               │  │ │
│  │  │ - id (PK)              │  - username, email                   │  │ │
│  │  │ - hashed_password      │  - role (admin/operator/analyst)     │  │ │
│  │  │ - full_name            │  - is_active, last_login             │  │ │
│  │  └────────────────────────┴───────────────────────────────────────┘  │ │
│  │                                                                        │ │
│  │  ┌────────────────────────────────────────────────────────────────┐  │ │
│  │  │                    Database Relationships                       │  │ │
│  │  │                                                                 │  │ │
│  │  │  DateTimeTable (1) ──< (M) VoltageTable                        │  │ │
│  │  │  DateTimeTable (1) ──< (M) CurrentTable                        │  │ │
│  │  │  DateTimeTable (1) ──< (M) FrequencyTable                      │  │ │
│  │  │  DateTimeTable (1) ──< (M) ActivePowerTable                    │  │ │
│  │  │  DateTimeTable (1) ──< (M) ReactivePowerTable                  │  │ │
│  │  │                                                                 │  │ │
│  │  │  All measurement tables use timestamp_id as Foreign Key        │  │ │
│  │  └────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                        │ │
│  │  ┌────────────────────────────────────────────────────────────────┐  │ │
│  │  │                      Indexing Strategy                          │  │ │
│  │  │  - Primary Keys: Clustered index on id                         │  │ │
│  │  │  - Foreign Keys: Index on timestamp_id for joins               │  │ │
│  │  │  - Timestamp: Index on DateTimeTable.timestamp for queries     │  │ │
│  │  │  - Simulation: Index on simulation_id for filtering            │  │ │
│  │  └────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                      Blob Storage (Future)                            │ │
│  │                      - ML model files                                 │ │
│  │                      - Generated reports (PDF, CSV)                   │ │
│  │                      - Historical data archives                       │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Across Tiers

### Request-Response Flow (Synchronous)

```
User Browser (Tier 1)
    │
    │ 1. User requests dashboard
    ▼
Django Frontend (Tier 1)
    │
    │ 2. HTTP GET /api/grid/data/latest
    ▼
FastAPI Backend (Tier 2)
    │
    │ 3. SQLAlchemy ORM query
    ▼
MySQL Database (Tier 3)
    │
    │ 4. SELECT latest records
    ▼
FastAPI Backend (Tier 2)
    │
    │ 5. Feature engineering (80+ features)
    │ 6. ML inference (16 models)
    │ 7. JSON response
    ▼
Django Frontend (Tier 1)
    │
    │ 8. Render dashboard with Chart.js
    ▼
User Browser (Tier 1)
```

### Real-time WebSocket Flow (Asynchronous)

```
Grid SCADA/IoT Device
    │
    │ 1. POST /api/grid/data
    ▼
FastAPI Backend (Tier 2)
    │
    ├─► 2a. Validate data (Pydantic)
    │
    ├─► 2b. Save to MySQL (Tier 3)
    │   └─► INSERT INTO DateTimeTable, VoltageTable, etc.
    │
    ├─► 2c. Feature engineering
    │   └─► Extract 80+ features
    │
    ├─► 2d. ML inference
    │   └─► Run 16 models
    │
    └─► 2e. Broadcast via WebSocket
        │
        ▼
    WebSocket Server (/ws/grid-data)
        │
        │ 3. Send JSON payload
        ▼
    Connected Clients (Tier 1)
        │
        │ 4. Update dashboard in real-time
        ▼
    Chart.js updates visualizations
```

---

## Component Details

### Tier 1: Presentation Layer (Django Frontend)

**Technology Stack**
- **Framework**: Django 5.2.8
- **Template Engine**: Django Templates (Jinja-like syntax)
- **Static Assets**: Chart.js, D3.js, Bootstrap, Tailwind
- **Database**: SQLite3 (session storage only, not for grid data)
- **Port**: 8000

**Key Features**
- Server-side rendering for SEO and initial page load
- Session-based authentication (cookies)
- AJAX calls to backend API for dynamic data
- WebSocket client for real-time updates
- Responsive design (mobile-friendly)

**File Structure**
```
frontend/
├── smartgrid/              # Django project
│   ├── settings.py         # Configuration
│   ├── urls.py             # URL routing
│   └── wsgi.py             # WSGI application
├── dashboard/              # Django app
│   ├── views.py            # View handlers
│   ├── urls.py             # App URL routing
│   ├── templates/          # HTML templates
│   │   ├── base.html       # Base template
│   │   ├── dashboard.html  # Main dashboard
│   │   └── graphs/         # Visualization pages
│   └── static/             # CSS, JS, images
└── db.sqlite3              # Local session DB
```

**URL Routes**
```python
# Django frontend routes
/                           # Landing page
/login/                     # Login page
/logout/                    # Logout
/dashboard/                 # Main dashboard
/dashboard/realtime-monitoring/     # ML Module 1
/dashboard/predictive-maintenance/  # ML Module 2
/dashboard/energy-flow/             # ML Module 3
/dashboard/decision-making/         # ML Module 4
/dashboard/simulator/               # Grid simulator UI
/dashboard/reports/                 # Report generation
```

**Backend API Integration**
```python
# views.py example
import requests

def dashboard_view(request):
    # Fetch latest grid data from backend API
    response = requests.get('http://localhost:8001/api/grid/data/latest')
    grid_data = response.json()

    return render(request, 'dashboard.html', {
        'grid_data': grid_data
    })
```

---

### Tier 2: Application Layer (FastAPI Backend)

**Technology Stack**
- **Framework**: FastAPI 0.104.1 (async web framework)
- **ORM**: SQLAlchemy 2.0.23
- **Validation**: Pydantic 2.5.0
- **ML Libraries**: TensorFlow, XGBoost, Prophet, scikit-learn
- **Server**: Uvicorn (ASGI)
- **Port**: 8001

**Key Features**
- Fully asynchronous (async/await)
- Automatic API documentation (Swagger UI at /docs)
- Type-safe with Pydantic schemas
- WebSocket support for real-time data
- Singleton pattern for ML model management

**File Structure**
```
backend/app/
├── main.py                 # FastAPI app entry point
├── database.py             # SQLAlchemy config
├── models/
│   └── db_models.py        # ORM models
├── routers/                # API endpoints
│   ├── auth.py             # /api/auth
│   ├── grid_data.py        # /api/grid
│   ├── websocket_router.py # /ws
│   ├── simulator.py        # /api/simulator
│   ├── ml_monitoring.py    # /api/ml/monitoring
│   ├── ml_maintenance.py   # /api/ml/maintenance
│   ├── ml_energy.py        # /api/ml/energy
│   └── ml_decision.py      # /api/ml/decision
├── schemas/                # Pydantic schemas
│   ├── grid_data.py
│   └── simulation.py
├── services/               # Business logic
│   ├── feature_engineering.py  # 80+ features
│   ├── model_manager.py        # ML model loader
│   ├── ml_inference_engine.py  # Inference orchestration
│   └── simulation_generator.py
├── ml_models/
│   └── trained/            # 16 trained models
└── utils/
    └── security.py         # JWT, RBAC
```

**API Endpoints**
```python
# main.py
from fastapi import FastAPI

app = FastAPI(title="REMHART Digital Twin API", version="1.0.0")

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(grid_data.router, prefix="/api/grid", tags=["Grid Data"])
app.include_router(simulator.router, prefix="/api/simulator", tags=["Simulator"])
app.include_router(ml_monitoring.router, prefix="/api/ml/monitoring", tags=["ML Monitoring"])
# ... more routers

@app.get("/")
async def root():
    return {"message": "REMHART Digital Twin API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

**Business Logic Layer**

```python
# Feature Engineering Service
class FeatureEngineer:
    def extract_features(self, grid_data: GridData) -> Dict:
        """Extract 80+ features from raw grid data"""
        features = {}

        # Voltage features (11)
        features['voltage_avg'] = (grid_data.voltage_a + grid_data.voltage_b + grid_data.voltage_c) / 3
        features['voltage_imbalance'] = calculate_imbalance(...)
        # ... 9 more voltage features

        # Current features (11)
        features['current_avg'] = ...
        # ... more features

        # Total: 80+ features
        return features

# ML Inference Engine
class MLInferenceEngine:
    def __init__(self):
        self.model_manager = ModelManager()  # Singleton

    async def run_all_modules(self, features: Dict) -> Dict:
        """Run inference across all 16 models"""
        results = {}

        # Module 1: Real-time Monitoring (4 models)
        results['monitoring'] = await self.run_monitoring_module(features)

        # Module 2: Predictive Maintenance (4 models)
        results['maintenance'] = await self.run_maintenance_module(features)

        # Module 3: Energy Flow (4 models)
        results['energy'] = await self.run_energy_module(features)

        # Module 4: Decision Making (4 models)
        results['decision'] = await self.run_decision_module(features)

        return results
```

**Dependency Injection**
```python
from fastapi import Depends
from sqlalchemy.orm import Session

# Database session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Route using dependency injection
@router.post("/data")
async def create_grid_data(
    data: GridDataCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Business logic here
    pass
```

---

### Tier 3: Data Layer (MySQL Database)

**Technology Stack**
- **Database**: MySQL 8.0+
- **ORM**: SQLAlchemy 2.0.23
- **Migration**: Alembic
- **Connection**: pymysql / mysqlclient

**Database Schema**

```sql
-- DateTimeTable (Central timestamp reference)
CREATE TABLE datetime_table (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME(6) NOT NULL,  -- Microsecond precision
    is_simulation BOOLEAN DEFAULT FALSE,
    simulation_id VARCHAR(255),
    simulation_name VARCHAR(255),
    simulation_scenario VARCHAR(255),
    INDEX idx_timestamp (timestamp),
    INDEX idx_simulation (simulation_id)
);

-- VoltageTable (3-phase voltage)
CREATE TABLE voltage_table (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp_id INT NOT NULL,
    phaseA FLOAT NOT NULL CHECK (phaseA >= 0 AND phaseA <= 300),
    phaseB FLOAT NOT NULL CHECK (phaseB >= 0 AND phaseB <= 300),
    phaseC FLOAT NOT NULL CHECK (phaseC >= 0 AND phaseC <= 300),
    average FLOAT,
    FOREIGN KEY (timestamp_id) REFERENCES datetime_table(id) ON DELETE CASCADE,
    INDEX idx_timestamp_fk (timestamp_id)
);

-- CurrentTable (3-phase current)
CREATE TABLE current_table (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp_id INT NOT NULL,
    phaseA FLOAT NOT NULL,
    phaseB FLOAT NOT NULL,
    phaseC FLOAT NOT NULL,
    average FLOAT,
    FOREIGN KEY (timestamp_id) REFERENCES datetime_table(id) ON DELETE CASCADE,
    INDEX idx_timestamp_fk (timestamp_id)
);

-- FrequencyTable (Grid frequency)
CREATE TABLE frequency_table (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp_id INT NOT NULL,
    frequency_value FLOAT NOT NULL CHECK (frequency_value >= 49.0 AND frequency_value <= 51.0),
    FOREIGN KEY (timestamp_id) REFERENCES datetime_table(id) ON DELETE CASCADE,
    INDEX idx_timestamp_fk (timestamp_id)
);

-- ActivePowerTable (Real power)
CREATE TABLE active_power_table (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp_id INT NOT NULL,
    phaseA FLOAT NOT NULL,
    phaseB FLOAT NOT NULL,
    phaseC FLOAT NOT NULL,
    total FLOAT,
    FOREIGN KEY (timestamp_id) REFERENCES datetime_table(id) ON DELETE CASCADE,
    INDEX idx_timestamp_fk (timestamp_id)
);

-- ReactivePowerTable (Reactive power)
CREATE TABLE reactive_power_table (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp_id INT NOT NULL,
    phaseA FLOAT NOT NULL,
    phaseB FLOAT NOT NULL,
    phaseC FLOAT NOT NULL,
    total FLOAT,
    FOREIGN KEY (timestamp_id) REFERENCES datetime_table(id) ON DELETE CASCADE,
    INDEX idx_timestamp_fk (timestamp_id)
);

-- SimulationMetadata (Simulation tracking)
CREATE TABLE simulation_metadata (
    id INT AUTO_INCREMENT PRIMARY KEY,
    simulation_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    scenario_type VARCHAR(50),
    parameters JSON,
    is_active BOOLEAN DEFAULT FALSE,
    status VARCHAR(50) DEFAULT 'created',
    created_by VARCHAR(255),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    total_points INT DEFAULT 0,
    INDEX idx_active (is_active),
    INDEX idx_scenario (scenario_type)
);

-- User (Authentication & RBAC)
CREATE TABLE user (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    role ENUM('admin', 'operator', 'analyst', 'viewer') DEFAULT 'viewer',
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_login DATETIME,
    INDEX idx_username (username),
    INDEX idx_email (email),
    INDEX idx_role (role)
);
```

**ORM Models (SQLAlchemy)**

```python
from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class DateTimeTable(Base):
    __tablename__ = 'datetime_table'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(6), nullable=False)
    is_simulation = Column(Boolean, default=False)
    simulation_id = Column(String(255), nullable=True)
    simulation_name = Column(String(255), nullable=True)
    simulation_scenario = Column(String(255), nullable=True)

    # Relationships
    voltage = relationship("VoltageTable", back_populates="datetime", cascade="all, delete-orphan")
    current = relationship("CurrentTable", back_populates="datetime", cascade="all, delete-orphan")
    frequency = relationship("FrequencyTable", back_populates="datetime", cascade="all, delete-orphan")
    active_power = relationship("ActivePowerTable", back_populates="datetime", cascade="all, delete-orphan")
    reactive_power = relationship("ReactivePowerTable", back_populates="datetime", cascade="all, delete-orphan")

class VoltageTable(Base):
    __tablename__ = 'voltage_table'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp_id = Column(Integer, ForeignKey('datetime_table.id', ondelete='CASCADE'), nullable=False)
    phaseA = Column(Float, nullable=False)
    phaseB = Column(Float, nullable=False)
    phaseC = Column(Float, nullable=False)
    average = Column(Float)

    # Relationship
    datetime = relationship("DateTimeTable", back_populates="voltage")
```

**Database Operations**

```python
# Create operation
async def create_grid_data(db: Session, data: GridDataCreate):
    # Create timestamp entry
    db_datetime = DateTimeTable(
        timestamp=datetime.utcnow(),
        is_simulation=data.is_simulation
    )
    db.add(db_datetime)
    db.flush()  # Get the ID

    # Create voltage entry
    db_voltage = VoltageTable(
        timestamp_id=db_datetime.id,
        phaseA=data.voltage_a,
        phaseB=data.voltage_b,
        phaseC=data.voltage_c,
        average=(data.voltage_a + data.voltage_b + data.voltage_c) / 3
    )
    db.add(db_voltage)

    # ... create other tables

    db.commit()
    db.refresh(db_datetime)
    return db_datetime

# Read operation
async def get_latest_grid_data(db: Session):
    result = db.query(DateTimeTable).order_by(DateTimeTable.timestamp.desc()).first()
    return result

# Query with joins
async def get_grid_data_with_measurements(db: Session, limit: int = 100):
    results = db.query(DateTimeTable)\
        .join(VoltageTable)\
        .join(CurrentTable)\
        .join(FrequencyTable)\
        .join(ActivePowerTable)\
        .join(ReactivePowerTable)\
        .order_by(DateTimeTable.timestamp.desc())\
        .limit(limit)\
        .all()
    return results
```

---

## Inter-Tier Communication

### Communication Protocols

| Tier 1 → Tier 2 | Tier 2 → Tier 3 |
|-----------------|-----------------|
| HTTP REST API | SQLAlchemy ORM |
| WebSocket (ws://) | MySQL Protocol |
| JSON payloads | SQL queries |
| JWT Bearer tokens | Connection pool |

### Data Formats

**API Request (Tier 1 → Tier 2)**
```json
POST /api/grid/data
{
  "voltage_a": 230.5,
  "voltage_b": 229.8,
  "voltage_c": 231.2,
  "current_a": 15.3,
  "current_b": 14.9,
  "current_c": 15.1,
  "frequency": 50.02,
  "active_power_a": 3526.5,
  "active_power_b": 3427.02,
  "active_power_c": 3491.12,
  "reactive_power_a": 150.0,
  "reactive_power_b": 145.0,
  "reactive_power_c": 148.0
}
```

**API Response (Tier 2 → Tier 1)**
```json
{
  "id": 12345,
  "timestamp": "2025-12-27T10:30:45.123456",
  "voltage": {
    "phaseA": 230.5,
    "phaseB": 229.8,
    "phaseC": 231.2,
    "average": 230.5
  },
  "current": { ... },
  "frequency": { "value": 50.02 },
  "power_quality": {
    "voltage_imbalance": 0.6,
    "thd": 2.1,
    "power_factor": 0.97
  },
  "ml_predictions": {
    "voltage_anomaly_detected": false,
    "grid_stability_score": 94.5,
    "equipment_failure_risk": 0.12
  }
}
```

**WebSocket Message (Tier 2 → Tier 1)**
```json
{
  "type": "grid_data_update",
  "timestamp": "2025-12-27T10:30:45.123456",
  "data": {
    "voltage_avg": 230.5,
    "current_avg": 15.1,
    "frequency": 50.02,
    "total_active_power": 10444.64,
    "grid_stability_score": 94.5
  }
}
```

---

## Security Layers

### Tier 1 (Frontend) Security
- **Session-based authentication**: Django session cookies
- **CSRF protection**: Django CSRF tokens
- **XSS protection**: Template auto-escaping
- **Input validation**: Client-side + server-side

### Tier 2 (Backend) Security
- **JWT authentication**: Bearer tokens in Authorization header
- **RBAC**: Role-based access control (4 roles)
- **CORS**: Configured allowed origins
- **Rate limiting**: TODO (future enhancement)
- **SQL injection prevention**: ORM parameterized queries

### Tier 3 (Database) Security
- **Password encryption**: Bcrypt hashing
- **Database encryption**: Transparent Data Encryption (TDE)
- **Access control**: MySQL user permissions
- **Backup encryption**: Encrypted backups

---

## Performance Characteristics

### Latency Targets

| Operation | Target Latency |
|-----------|----------------|
| API request (simple GET) | <50ms |
| API request (with ML inference) | <150ms |
| WebSocket message delivery | <10ms |
| Database query (indexed) | <20ms |
| Database write operation | <30ms |
| Page load (initial) | <2s |

### Throughput Capacity

| Metric | Capacity |
|--------|----------|
| API requests/second | 500+ (single instance) |
| WebSocket connections | 1000+ concurrent |
| Database writes/second | 200+ (single instance) |
| ML inferences/second | 100+ (16 models per request) |

### Scalability Approach

- **Tier 1 (Frontend)**: Horizontal scaling with load balancer
- **Tier 2 (Backend)**: Horizontal scaling with multiple FastAPI instances
- **Tier 3 (Database)**: Vertical scaling + read replicas

---

## Deployment Configuration

### Development Environment
```yaml
# docker-compose.yml
version: '3.8'

services:
  frontend:
    build: ./frontend
    ports:
      - "8000:8000"
    environment:
      - BACKEND_URL=http://backend:8001
    depends_on:
      - backend

  backend:
    build: ./backend
    ports:
      - "8001:8001"
    environment:
      - DATABASE_URL=mysql://user:pass@mysql:3306/remhart_db
    depends_on:
      - mysql

  mysql:
    image: mysql:8.0
    ports:
      - "3306:3306"
    environment:
      - MYSQL_ROOT_PASSWORD=rootpass
      - MYSQL_DATABASE=remhart_db
    volumes:
      - mysql_data:/var/lib/mysql

volumes:
  mysql_data:
```

### Production Environment
- **Tier 1**: Gunicorn with 4+ workers
- **Tier 2**: Uvicorn with 4+ workers
- **Tier 3**: Managed MySQL with replication

---

## Monitoring & Observability

### Application Metrics

**Tier 1 (Frontend)**
- Page load time
- User session count
- WebSocket connection count

**Tier 2 (Backend)**
- API request rate
- Response time (p50, p95, p99)
- Error rate (4xx, 5xx)
- ML inference latency
- Active database connections

**Tier 3 (Database)**
- Query execution time
- Slow query count
- Connection pool utilization
- Disk I/O
- Table size growth

---

## Conclusion

The REMHART Digital Twin 3-tier architecture provides:

✅ **Separation of Concerns**: Clear boundaries between presentation, logic, and data
✅ **Scalability**: Each tier can scale independently
✅ **Maintainability**: Modular design with well-defined interfaces
✅ **Security**: Multi-layered security approach
✅ **Performance**: Optimized for real-time grid monitoring (<150ms latency)
✅ **Extensibility**: Easy to add new features and ML models

This architecture supports both current requirements and future growth of the smart grid digital twin platform.
