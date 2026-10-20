from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_prediction_endpoint():
    payload = {
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

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in [0, 1]
    assert result["label"] in [
        "disease_present",
        "disease_not_detected",
    ]

    probabilities = result["class_probabilities"]

    assert set(probabilities.keys()) == {"0", "1"}
    assert all(0 <= p <= 1 for p in probabilities.values())
    assert abs(sum(probabilities.values()) - 1.0) < 1e-6

    assert 0 <= result["confidence"] <= 1


def test_prediction_rejects_invalid_input():
    payload = {
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

    payload["age"] = 999

    response = client.post("/predict", json=payload)

    assert response.status_code == 422