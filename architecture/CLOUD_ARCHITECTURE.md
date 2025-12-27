# REMHART Digital Twin - Cloud Architecture

## Overview

The REMHART Smart Grid Digital Twin system is designed for cloud-native deployment with microservices architecture, enabling scalable real-time monitoring and AI-powered decision support for electrical grid infrastructure.

---

## Cloud Deployment Architecture

### Target Cloud Platform: Azure / AWS / Google Cloud

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLOUD INFRASTRUCTURE                             │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    Load Balancer / API Gateway                     │ │
│  │              (SSL/TLS Termination, Rate Limiting)                  │ │
│  └───────────────────────┬──────────────────┬────────────────────────┘ │
│                          │                  │                           │
│  ┌───────────────────────▼────┐  ┌─────────▼──────────────────────┐   │
│  │   Frontend Service         │  │   Backend API Service           │   │
│  │   (Django Container)       │  │   (FastAPI Container)           │   │
│  │   Port: 8000               │  │   Port: 8001                    │   │
│  │   Instances: Auto-scaled   │  │   Instances: Auto-scaled        │   │
│  │   - Web UI                 │  │   - REST API                    │   │
│  │   - Dashboard              │  │   - WebSocket Server            │   │
│  │   - User Sessions          │  │   - ML Inference Engine         │   │
│  └────────────────────────────┘  │   - Grid Data Processing        │   │
│                                   │   - Simulation Engine           │   │
│                                   └──────────┬──────────────────────┘   │
│                                              │                           │
│  ┌───────────────────────────────────────────▼────────────────────────┐ │
│  │                    Managed Database Service                         │ │
│  │                    (MySQL / Azure SQL / RDS)                        │ │
│  │                    - Grid measurement data                          │ │
│  │                    - User accounts & RBAC                           │ │
│  │                    - Simulation metadata                            │ │
│  │                    - Automated backups, HA                          │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    Blob Storage / Object Storage                    │ │
│  │                    (Azure Blob / S3 / Cloud Storage)                │ │
│  │                    - ML model files (.pkl, .h5)                     │ │
│  │                    - Generated reports (PDF, CSV)                   │ │
│  │                    - Logs and analytics                             │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    Message Queue / Event Bus                        │ │
│  │                    (Azure Service Bus / SQS / Pub/Sub)              │ │
│  │                    - Real-time grid data streaming                  │ │
│  │                    - ML inference task queue                        │ │
│  │                    - WebSocket event distribution                   │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    Monitoring & Observability                       │ │
│  │                    (Azure Monitor / CloudWatch / Stackdriver)       │ │
│  │                    - Application Performance Monitoring             │ │
│  │                    - Log aggregation                                │ │
│  │                    - Metrics & alerts                               │ │
│  │                    - Distributed tracing                            │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    Security Services                                │ │
│  │                    - Key Vault (JWT secrets, DB passwords)          │ │
│  │                    - Identity & Access Management (IAM)             │ │
│  │                    - DDoS Protection                                │ │
│  │                    - Web Application Firewall (WAF)                 │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└───────────────────────────────────────────────────────────────────────────┘
         ▲                                                    ▲
         │                                                    │
    ┌────┴─────────┐                              ┌──────────┴────────┐
    │  Grid SCADA  │                              │  External Users   │
    │  Data Source │                              │  (Web Browsers)   │
    │  (IoT/Edge)  │                              │  Operators/Admins │
    └──────────────┘                              └───────────────────┘
```

---

## Cloud Service Mapping

### Compute Services

| Component | Azure | AWS | Google Cloud |
|-----------|-------|-----|--------------|
| **Frontend Container** | Azure App Service / Container Instances | ECS Fargate / App Runner | Cloud Run / GKE |
| **Backend Container** | Azure App Service / Container Instances | ECS Fargate / App Runner | Cloud Run / GKE |
| **Orchestration** | Azure Kubernetes Service (AKS) | Elastic Kubernetes Service (EKS) | Google Kubernetes Engine (GKE) |
| **Auto-scaling** | VMSS / App Service Plan | Auto Scaling Groups | Managed Instance Groups |

### Data Services

| Component | Azure | AWS | Google Cloud |
|-----------|-------|-----|--------------|
| **Primary Database** | Azure Database for MySQL | RDS for MySQL | Cloud SQL for MySQL |
| **Cache** | Azure Cache for Redis | ElastiCache | Memorystore |
| **Object Storage** | Azure Blob Storage | Amazon S3 | Cloud Storage |
| **Message Queue** | Azure Service Bus | Amazon SQS | Cloud Pub/Sub |
| **Streaming** | Azure Event Hubs | Amazon Kinesis | Cloud Dataflow |

### ML/AI Services

| Component | Azure | AWS | Google Cloud |
|-----------|-------|-----|--------------|
| **ML Model Hosting** | Azure ML | SageMaker | Vertex AI |
| **Model Registry** | Azure ML Model Registry | SageMaker Model Registry | Vertex AI Model Registry |
| **Inference** | Azure ML Endpoints | SageMaker Inference | Vertex AI Predictions |

### Security & Identity

| Component | Azure | AWS | Google Cloud |
|-----------|-------|-----|--------------|
| **Secrets Management** | Azure Key Vault | AWS Secrets Manager | Secret Manager |
| **Identity** | Azure AD / Entra ID | AWS IAM / Cognito | Cloud IAM |
| **API Management** | Azure API Management | API Gateway | Apigee / API Gateway |

### Monitoring & Observability

| Component | Azure | AWS | Google Cloud |
|-----------|-------|-----|--------------|
| **Application Monitoring** | Application Insights | CloudWatch / X-Ray | Cloud Monitoring / Trace |
| **Log Management** | Azure Monitor Logs | CloudWatch Logs | Cloud Logging |
| **Metrics** | Azure Monitor Metrics | CloudWatch Metrics | Cloud Monitoring |
| **Alerting** | Azure Monitor Alerts | CloudWatch Alarms | Cloud Monitoring Alerts |

---

## Container Architecture

### Docker Containerization

**Frontend Container (Django)**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY frontend/ .
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "smartgrid.wsgi:application"]
```

**Backend Container (FastAPI)**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ .
EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "4"]
```

### Kubernetes Deployment

**Pod Architecture**
- **Frontend Pods**: 2-10 replicas (auto-scaled based on CPU/memory)
- **Backend Pods**: 3-20 replicas (auto-scaled based on request rate)
- **Horizontal Pod Autoscaler (HPA)**: Target 70% CPU utilization
- **Resource Limits**:
  - Frontend: 512MB RAM, 0.5 CPU cores per pod
  - Backend: 2GB RAM, 1.0 CPU cores per pod (ML inference requires more memory)

---

## Network Architecture

### Ingress & Load Balancing

```
Internet Traffic
      │
      ▼
┌─────────────────┐
│  API Gateway    │  ← SSL/TLS Termination (HTTPS)
│  Load Balancer  │  ← DDoS Protection
│                 │  ← Rate Limiting: 1000 req/min per IP
└────────┬────────┘
         │
    ┌────┴────┐
    │  Rules  │
    └────┬────┘
         │
    ┌────┴──────────────────────┐
    │                           │
    ▼                           ▼
/dashboard/*              /api/*, /ws/*
Frontend Service          Backend Service
(Port 8000)               (Port 8001)
```

### Service Mesh (Optional - for microservices expansion)

- **Istio / Linkerd**: For service-to-service communication
- **mTLS**: Mutual TLS between services
- **Circuit Breaking**: Prevent cascade failures
- **Observability**: Distributed tracing across services

---

## Data Flow in Cloud

### Real-time Data Ingestion Pipeline

```
Grid SCADA/IoT Device
        │
        │ MQTT/HTTP POST
        ▼
┌─────────────────┐
│  IoT Hub/       │
│  Event Hub      │  ← Ingestion: 10,000+ events/sec
└────────┬────────┘
         │
         │ Event Stream
         ▼
┌─────────────────┐
│  Backend API    │  ← Validation & Parsing
│  Service        │  ← Feature Engineering (80+ features)
└────────┬────────┘
         │
    ┌────┴─────┐
    │          │
    ▼          ▼
┌────────┐  ┌─────────────┐
│ MySQL  │  │ ML Inference│  ← 16 Models
│ Write  │  │ Engine      │  ← <100ms latency
└────────┘  └──────┬──────┘
                   │
                   │ Predictions
                   ▼
            ┌──────────────┐
            │  WebSocket   │  ← Real-time broadcast
            │  Server      │  ← to connected clients
            └──────┬───────┘
                   │
                   ▼
            ┌──────────────┐
            │  Frontend    │
            │  Dashboard   │
            └──────────────┘
```

---

## Scalability & Performance

### Auto-scaling Configuration

**Horizontal Pod Autoscaler (HPA)**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Database Scaling

- **Read Replicas**: 2-3 read replicas for analytics queries
- **Connection Pooling**: SQLAlchemy pool (min=5, max=20 per pod)
- **Caching**: Redis for frequently accessed data (grid status, latest readings)
- **Partitioning**: Time-series partitioning by month for historical data

### ML Model Scaling

- **Model Caching**: Models loaded once per pod (Singleton pattern)
- **Batch Inference**: Batch multiple requests for efficiency
- **GPU Acceleration**: Optional GPU nodes for LSTM/Neural Network models
- **Model Registry**: Centralized model versioning and deployment

---

## High Availability & Disaster Recovery

### Availability Zones

- **Multi-AZ Deployment**: Services deployed across 3 availability zones
- **Database**: Multi-AZ with automatic failover
- **Target SLA**: 99.95% uptime (4.38 hours downtime/year)

### Backup & Recovery

- **Database Backups**: Automated daily backups, 30-day retention
- **Point-in-Time Recovery (PITR)**: 5-minute granularity
- **Blob Storage**: Geo-redundant storage for ML models and reports
- **Recovery Time Objective (RTO)**: < 1 hour
- **Recovery Point Objective (RPO)**: < 5 minutes

### Health Checks

```python
# Backend health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database": check_db_connection(),
        "ml_models_loaded": model_manager.is_loaded(),
        "timestamp": datetime.utcnow()
    }
```

---

## Security Architecture

### Network Security

- **VPC/VNet Isolation**: Private subnets for backend and database
- **Security Groups**: Restrict traffic to necessary ports only
- **WAF Rules**: OWASP Top 10 protection
- **DDoS Protection**: Cloud-native DDoS mitigation

### Application Security

- **Authentication**: JWT tokens with 1-hour expiration
- **Authorization**: Role-Based Access Control (RBAC) - 4 roles
- **Encryption in Transit**: TLS 1.3 for all communications
- **Encryption at Rest**: Database and blob storage encryption
- **Secrets Management**: Key Vault for all credentials
- **API Rate Limiting**: 1000 requests/minute per user

### Compliance

- **Data Residency**: Configurable region deployment
- **Audit Logging**: All API calls logged for compliance
- **GDPR**: User data deletion capabilities
- **SOC 2**: Cloud provider compliance inheritance

---

## Cost Optimization

### Resource Sizing

| Component | vCPU | Memory | Storage | Monthly Cost (Est.) |
|-----------|------|--------|---------|---------------------|
| **Frontend Pods** (avg 3 replicas) | 1.5 | 1.5 GB | - | $50 |
| **Backend Pods** (avg 5 replicas) | 5.0 | 10 GB | - | $150 |
| **MySQL Database** (Standard tier) | 2 | 8 GB | 100 GB | $200 |
| **Redis Cache** (Basic tier) | - | 1 GB | - | $30 |
| **Blob Storage** (100 GB) | - | - | 100 GB | $5 |
| **Load Balancer** | - | - | - | $25 |
| **Monitoring** | - | - | - | $20 |
| **Total** | | | | **~$480/month** |

### Cost Optimization Strategies

1. **Reserved Instances**: 40% savings on database and compute
2. **Spot Instances**: Use for batch ML training jobs
3. **Auto-scaling**: Scale down during off-peak hours
4. **Data Lifecycle**: Archive historical data to cold storage after 90 days
5. **CDN**: Cache static assets to reduce compute load

---

## Monitoring & Alerting

### Key Metrics

**Application Metrics**
- Request rate (req/sec)
- Response time (p50, p95, p99)
- Error rate (4xx, 5xx)
- WebSocket connections (active count)

**ML Metrics**
- Inference latency (ms)
- Model prediction accuracy (daily eval)
- Feature engineering time (ms)

**Infrastructure Metrics**
- CPU utilization (%)
- Memory utilization (%)
- Disk I/O (IOPS)
- Network throughput (Mbps)

### Alert Thresholds

| Alert | Threshold | Action |
|-------|-----------|--------|
| **High Error Rate** | >5% 5xx errors | Page on-call engineer |
| **High Latency** | p95 >1000ms | Auto-scale pods |
| **Database Connection** | >90% pool usage | Increase connection pool |
| **ML Inference Failure** | >10 failures/min | Reload models, fallback mode |
| **Disk Space** | >85% used | Trigger data archival |

---

## Deployment Pipeline (CI/CD)

### GitOps Workflow

```
Developer Push to Git
        │
        ▼
┌───────────────┐
│ GitHub Actions│  ← Automated CI/CD
│ / Azure DevOps│
└───────┬───────┘
        │
   ┌────┴────┐
   │ Stages  │
   └────┬────┘
        │
        ├──► 1. Lint & Test (pytest, mypy)
        ├──► 2. Build Docker Images
        ├──► 3. Push to Container Registry
        ├──► 4. Deploy to Staging Environment
        ├──► 5. Run Integration Tests
        ├──► 6. Deploy to Production (approval required)
        └──► 7. Health Check & Rollback if failed
```

### Blue-Green Deployment

- **Zero-downtime deployment**: Blue-green strategy
- **Canary releases**: 10% traffic → 50% → 100%
- **Automatic rollback**: If error rate >5% or latency >2x baseline

---

## API Gateway Configuration

### Endpoints Exposed

```
https://api.remhart-digitaltwin.com/

├── /api/auth/              (Authentication)
├── /api/grid/              (Grid data operations)
├── /api/simulator/         (Simulation control)
├── /api/ml/                (ML inference endpoints)
│   ├── /monitoring/        (Real-time monitoring)
│   ├── /maintenance/       (Predictive maintenance)
│   ├── /energy/            (Energy flow)
│   └── /decision/          (Decision support)
├── /api/reports/           (Report generation)
└── /ws/grid-data           (WebSocket real-time stream)
```

### Rate Limiting

- **Public endpoints**: 100 req/min per IP
- **Authenticated users**: 1000 req/min per user
- **Admin users**: 5000 req/min
- **WebSocket**: Max 1000 concurrent connections per region

---

## Edge Computing Integration (Optional)

### Hybrid Cloud-Edge Architecture

For low-latency requirements at substations:

```
┌────────────────────────────────┐
│  Edge Device (At Substation)   │
│  ┌──────────────────────────┐  │
│  │  Lightweight ML Models   │  │  ← Anomaly detection
│  │  (TensorFlow Lite)       │  │  ← Local processing
│  └──────────────────────────┘  │
│  ┌──────────────────────────┐  │
│  │  Edge Data Buffer        │  │  ← Store-and-forward
│  │  (Time-series DB)        │  │  ← Offline capability
│  └──────────────────────────┘  │
└────────────┬───────────────────┘
             │ Cellular/Fiber
             ▼
     ┌───────────────┐
     │  Cloud IoT    │
     │  Gateway      │
     └───────┬───────┘
             │
             ▼
     ┌───────────────┐
     │  Full Cloud   │
     │  Backend      │
     └───────────────┘
```

---

## Recommended Cloud Architecture: Azure-Specific

### Azure Services Stack

```
┌─────────────────────────────────────────────────────────────┐
│                      Azure Front Door                        │  ← Global CDN & WAF
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              Azure Application Gateway                       │  ← Regional load balancer
└────────────────┬────────────────────┬───────────────────────┘
                 │                    │
       ┌─────────▼─────────┐   ┌─────▼──────────────────┐
       │ Azure App Service │   │ Azure Container Apps   │
       │ (Frontend)        │   │ (Backend FastAPI)      │
       └─────────┬─────────┘   └─────┬──────────────────┘
                 │                    │
                 └────────┬───────────┘
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
   ┌────▼──────────────┐        ┌──────────▼─────────────┐
   │ Azure Database    │        │ Azure Blob Storage     │
   │ for MySQL         │        │ (ML models, reports)   │
   │ (Flexible Server) │        └────────────────────────┘
   └───────────────────┘
        │
   ┌────▼──────────────┐
   │ Azure Cache       │
   │ for Redis         │
   └───────────────────┘
```

### Azure-Specific Features

- **Azure Monitor**: Unified observability platform
- **Azure Key Vault**: Secrets and certificate management
- **Azure Service Bus**: Message queue for event-driven architecture
- **Azure Machine Learning**: ML model training and deployment
- **Azure Digital Twins** (Future): DTDL-based digital twin platform integration

---

## Migration Path (On-Premises → Cloud)

### Phase 1: Lift-and-Shift
- Deploy existing containers to cloud VMs
- Migrate MySQL database to managed service
- Set up VPN for hybrid connectivity

### Phase 2: Cloud Optimization
- Implement auto-scaling
- Add managed services (Redis cache, blob storage)
- Set up CI/CD pipelines

### Phase 3: Cloud-Native
- Implement serverless functions for batch jobs
- Add event-driven architecture with message queues
- Integrate cloud-native ML services

### Phase 4: Advanced Features
- Azure Digital Twins integration
- Multi-region deployment
- Edge computing for substations

---

## Conclusion

The REMHART Digital Twin cloud architecture provides:

✅ **Scalability**: Auto-scaling to handle 10,000+ events/second
✅ **High Availability**: 99.95% uptime with multi-AZ deployment
✅ **Security**: Enterprise-grade security with encryption, WAF, and RBAC
✅ **Performance**: <100ms ML inference latency, real-time WebSocket updates
✅ **Cost-Effective**: ~$480/month for small deployment, scales with usage
✅ **Flexibility**: Cloud-agnostic design (Azure/AWS/GCP)

This architecture supports the transition from prototype to production-grade smart grid monitoring system.
