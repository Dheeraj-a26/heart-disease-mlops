from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "heart_disease_clean.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "heart_disease_model.joblib"

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

SAMPLE = {
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 130,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.2,
    "slope": 2,
    "ca": 0,
    "thal": 3,
}


def test_clean_dataset_exists_and_has_expected_columns():
    assert DATA_PATH.exists(), "Cleaned dataset is missing."

    df = pd.read_csv(DATA_PATH)

    assert not df.empty
    assert set(FEATURES + ["target"]).issubset(df.columns)
    assert set(df["target"].dropna().unique()).issubset({0, 1})


def test_saved_model_exists():
    assert MODEL_PATH.exists(), "Saved model is missing."


def test_saved_model_is_a_pipeline():
    model = joblib.load(MODEL_PATH)

    assert isinstance(model, Pipeline)


def test_model_returns_binary_prediction():
    model = joblib.load(MODEL_PATH)

    sample_df = pd.DataFrame([SAMPLE], columns=FEATURES)
    prediction = model.predict(sample_df)

    assert len(prediction) == 1
    assert int(prediction[0]) in {0, 1}


def test_prediction_probabilities_are_valid():
    model = joblib.load(MODEL_PATH)

    sample_df = pd.DataFrame([SAMPLE], columns=FEATURES)
    probabilities = model.predict_proba(sample_df)

    assert probabilities.shape == (1, 2)
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.isclose(probabilities.sum(), 1.0)


def test_model_handles_missing_feature_values():
    model = joblib.load(MODEL_PATH)

    sample_df = pd.DataFrame([SAMPLE], columns=FEATURES)
    sample_df.loc[0, "chol"] = np.nan
    sample_df.loc[0, "thal"] = np.nan

    prediction = model.predict(sample_df)

    assert len(prediction) == 1
    assert int(prediction[0]) in {0, 1}