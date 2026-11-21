# Group G212 --- MP053: Hostel Mess Food Waste (Indore)

**Course**: SCSE3040 Machine Learning Operations (MLOps)  
**Problem ID**: MP053  
**Client**: Indore University Hostel Office  
**Faculty**: Saurabh Shrivastava | **Batch**: 55  

## Team Members
- **S24CSEU2521** --- Vishesh Aggarwal (Data & Model Lead)
- **S24CSEU0842** --- Deepam Pathak (Service Lead)
- **S24CSEU2534** --- Vabhravi Vashishth (Ops Lead)

---

## 1. Project Overview
The Indore University Hostel Office seeks to predict after-meal food waste (`waste_kg`) to optimize batch meal cooking across three hostels: Boys Hostel 1, Boys Hostel 2, and Girls Hostel. This is an end-to-end MLOps pipeline covering:
1. **Data Versioning & Quality Control**: DVC tracking, data anomaly detection (sentinel values, stuck sensors, spelling variants, target typos).
2. **Experiment Tracking & Model Registry**: MLflow with SQLite backend, artifact versioning, alias tagging (`@champion`).
3. **Serving & Validation**: FastAPI REST API (`POST /predict`, `GET /health`) with strict Pydantic input boundary rules.
4. **Drift Monitoring & Promotion**: 10-bin Population Stability Index (PSI), group MAE slicing on post-deployment data (`later.csv`), and strict automated promotion gating (>= 8% MAE improvement).
5. **Production Packaging**: Docker containerization (`mp053-api:1.0`), Kubernetes manifests (`mp053-api`, `mp053-svc`), and GitHub Actions CI.

---

## 2. Repository Layout
```
├── README.md                           # Group, problem, and member details
├── requirements.txt                    # Pinned production dependencies
├── params.yaml                         # Central pipeline configuration (read via yaml.safe_load)
├── src/                                # Modular python package
│   └── mess_waste/
│       ├── __init__.py
│       ├── data.py                     # Data checks, cleaning & pipeline loading
│       ├── features.py                 # Feature definitions & preprocessing pipelines
│       ├── model.py                    # Training & evaluation routines
│       └── drift.py                    # PSI and subgroup MAE evaluation
├── tests/                              # Pytest test suite (unit + TestClient)
│   ├── test_data.py
│   └── test_api.py
├── data/
│   └── raw/
│       ├── train.csv.dvc               # DVC tracking for raw training data
│       ├── later.csv.dvc               # DVC tracking for monitoring data
│       └── holdout_features.csv        # Holdout features for final evaluation
├── .dvc/                               # DVC configuration
├── mlflow.db                           # SQLite MLflow experiment and registry database
├── app/
│   └── main.py                         # FastAPI serving application with JSON logging
├── models/
│   └── model.joblib                    # Serialized production champion model
├── reports/
│   └── findings.yaml                   # Verified findings and evaluation metrics
├── predictions/
│   └── holdout_predictions.csv         # Predictions for final holdout dataset
├── Dockerfile                          # Container specification
├── .dockerignore                       # Build exclusion rules
├── k8s/                                # Kubernetes manifests
│   ├── deployment.yaml
│   └── service.yaml
└── .github/
    └── workflows/
        └── ci.yml                      # GitHub Actions continuous integration pipeline
```

---

## 3. Quickstart & Usage

### Setup Environment
```bash
pip install -r requirements.txt
```

### Reproduce Training & Registration
```bash
python -m src.mess_waste.train
```

### Run Service Locally
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
