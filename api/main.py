# api/main.py
from xmlrpc import client

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pickle
import numpy as np
import time
import os

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection — MLflow managed model",
    version="1.0.0"
)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

# ── Load Model + Scaler at Startup ────────────────────────────────────────────
# REPLACE the load at startup with this:
try:
    model = mlflow.sklearn.load_model("models:/fraud-detector@production")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    MODEL_LOADED = True
    print("Model loaded successfully.")
except Exception as e:
    MODEL_LOADED = False
    print(f"Model load failed: {e}")


# ── Schemas ───────────────────────────────────────────────────────────────────
class TransactionRequest(BaseModel):
    features: list[float]   # V1–V28, exactly 28 values
    amount: float
    time: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": [
                    -1.3598, -0.0728, 2.5363, 1.3782, -0.3383,
                    0.4624,  0.2396, 0.0987, 0.3638, 0.0908,
                    -0.5516, -0.6178, -0.9914, -0.3112, 1.4682,
                    -0.4704,  0.2080, 0.0258, 0.4040, 0.2514,
                    -0.0183,  0.2778, -0.1105, 0.0669, 0.1285,
                    -0.1891,  0.1336, -0.0211
                ],
                "amount": 149.62,
                "time": 0.0
            }
        }
    }


class FraudResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    latency_ms: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    mlflow_uri: str


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_risk_level(proba: float) -> str:
    if proba < 0.3:
        return "LOW"
    elif proba < 0.7:
        return "MEDIUM"
    return "HIGH"


# REPLACE get_production_version() with this:
def get_production_version() -> str:
    try:
        client = MlflowClient()
        v = client.get_model_version_by_alias("fraud-detector", "production")
        return f"v{v.version}"
    except Exception:
        return "unknown"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Fraud Detection API — visit /docs for interactive API"}


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if MODEL_LOADED else "degraded",
        model_loaded=MODEL_LOADED,
        mlflow_uri=MLFLOW_URI
    )


@app.post("/predict", response_model=FraudResponse)
async def predict(transaction: TransactionRequest):
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check MLflow connection."
        )

    if len(transaction.features) != 28:
        raise HTTPException(
            status_code=422,
            detail=f"Expected 28 features (V1-V28), got {len(transaction.features)}"
        )

    start = time.time()

    # Scale amount and time — same scaler used in training
    amount_scaled = scaler.transform([[transaction.amount]])[0][0]
    time_scaled = scaler.transform([[transaction.time]])[0][0]

    # Build full feature vector: V1-V28 + Amount_scaled + Time_scaled
    feature_vector = np.array(
        transaction.features + [amount_scaled, time_scaled]
    ).reshape(1, -1)

    pred = model.predict(feature_vector)[0]
    proba = model.predict_proba(feature_vector)[0][1]

    latency = (time.time() - start) * 1000

    return FraudResponse(
        transaction_id=f"txn_{int(time.time() * 1000)}",
        is_fraud=bool(pred == 1),
        fraud_probability=round(float(proba), 4),
        risk_level=get_risk_level(proba),
        latency_ms=round(latency, 2),
        model_version=get_production_version()
    )


@app.get("/model-info")
async def model_info():
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")

    client = MlflowClient()
    try:
        # REPLACE the /model-info endpoint's client call with this:
        latest = client.get_model_version_by_alias("fraud-detector", "production")
        run = client.get_run(latest.run_id)
        return {
            "model_name":        "fraud-detector",
            "version":           latest.version,
            "stage":             "Production",
            "sampling_strategy": run.data.params.get("sampling_strategy"),
            "model_type":        run.data.params.get("model_type"),
            "metrics": {
                "recall_fraud":    run.data.metrics.get("recall_fraud"),
                "precision_fraud": run.data.metrics.get("precision_fraud"),
                "roc_auc":         run.data.metrics.get("roc_auc"),
                "fraud_caught":    run.data.metrics.get("fraud_caught"),
                "fraud_missed":    run.data.metrics.get("fraud_missed"),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/threshold-info")
async def threshold_info():
    """
    Explains the precision-recall tradeoff.
    Useful talking point in interviews.
    """
    return {
        "current_threshold": 0.5,
        "model_recall":      0.8673,
        "model_precision":   0.7658,
        "explanation": (
            "Lowering threshold catches more fraud (higher recall) "
            "but increases false alarms (lower precision). "
            "For fraud detection, recall is prioritised — "
            "missing real fraud is more costly than a false alarm."
        ),
        "tradeoffs": {
            "threshold_0.3": "Higher recall, more false alarms — use when fraud cost is very high",
            "threshold_0.5": "Current default — balanced",
            "threshold_0.7": "Fewer false alarms, misses more fraud — use when review capacity is limited"
        }
    }
