from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
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
    version="1.0.0",
)


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


@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict(input_data: HeartDiseaseInput):
    try:
        # Convert input JSON into the DataFrame expected by the model.
        input_df = pd.DataFrame([input_data.model_dump()])

        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]

        return {
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

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid prediction input: {exc}",
        ) from exc