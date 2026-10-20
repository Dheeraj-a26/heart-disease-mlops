from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


def main():
    model = joblib.load(MODEL_PATH)

    # Illustrative input only; not a real patient record.
    sample = pd.DataFrame(
        [
            {
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
        ],
        columns=FEATURES,
    )

    prediction = int(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]

    print("Prediction:", prediction)
    print(
        "Predicted class:",
        "Disease present in dataset" if prediction == 1
        else "No disease recorded in dataset",
    )
    print(f"Probability for class 0: {probabilities[0]:.4f}")
    print(f"Probability for class 1: {probabilities[1]:.4f}")


if __name__ == "__main__":
    main()
