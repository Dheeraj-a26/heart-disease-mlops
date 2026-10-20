from pathlib import Path
from time import perf_counter
from threading import Lock

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "heart_disease_model.joblib"

# Load the trained model once when the API starts
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at {MODEL_PATH}. "
        "Run python src/train.py first."
    )

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "An ML inference API using a trained heart disease "
        "classification model. Predictions are for educational "
        "purposes and are not medical diagnoses."
    ),
    version="1.1.0",
)

# Basic in-memory monitoring metrics
metrics_lock = Lock()

metrics = {
    "total_requests": 0,
    "successful_predictions": 0,
    "failed_predictions": 0,
    "prediction_response_time_ms_total": 0.0,
    "prediction_response_time_ms_last": 0.0,
}


class HeartDiseaseInput(BaseModel):
    age: float = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    trestbps: float = Field(gt=0, le=300)
    chol: float = Field(gt=0, le=1000)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(gt=0, le=300)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0, le=10)
    slope: int = Field(ge=1, le=3)
    ca: float = Field(ge=0, le=3)
    thal: float = Field(ge=3, le=7)


@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Count API requests and prediction failures."""
    with metrics_lock:
        metrics["total_requests"] += 1

    try:
        response = await call_next(request)
    except Exception:
        if request.url.path == "/predict":
            with metrics_lock:
                metrics["failed_predictions"] += 1
        raise

    if request.url.path == "/predict" and response.status_code >= 400:
        with metrics_lock:
            metrics["failed_predictions"] += 1

    return response


@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API is running",
        "docs": "/docs",
        "metrics": "/metrics",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.get("/metrics")
def get_metrics():
    """Return basic API monitoring metrics."""
    with metrics_lock:
        result = metrics.copy()

    result["average_prediction_response_time_ms"] = round(
        result["prediction_response_time_ms_total"]
        / result["successful_predictions"],
        2,
    ) if result["successful_predictions"] > 0 else 0.0

    result["prediction_response_time_ms_total"] = round(
        result["prediction_response_time_ms_total"], 2
    )
    result["prediction_response_time_ms_last"] = round(
        result["prediction_response_time_ms_last"], 2
    )

    return result


@app.post("/predict")
def predict(input_data: HeartDiseaseInput):
    start_time = perf_counter()

    try:
        input_df = pd.DataFrame([input_data.model_dump()])

        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]

        result = {
            "prediction": prediction,
            "label": (
                "disease_present"
                if prediction == 1
                else "disease_not_detected"
            ),
            "confidence": round(float(max(probabilities)), 4),
            "class_probabilities": {
                str(index): round(float(probability), 4)
                for index, probability in enumerate(probabilities)
            },
            "disclaimer": (
                "Educational model output only; not a medical diagnosis."
            ),
        }

        with metrics_lock:
            metrics["successful_predictions"] += 1

        return result

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid prediction input: {exc}",
        ) from exc

    finally:
        elapsed_ms = (perf_counter() - start_time) * 1000

        with metrics_lock:
            metrics["prediction_response_time_ms_total"] += elapsed_ms
            metrics["prediction_response_time_ms_last"] = elapsed_ms