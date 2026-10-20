# Heart Disease Classification — MLOps Project

## 1. Project Overview

This project implements an end-to-end Machine Learning Operations (MLOps) pipeline for heart disease classification using the Cleveland Heart Disease dataset.

The project covers data acquisition, preprocessing, exploratory data analysis (EDA), model development, experiment tracking, model packaging, automated testing, Docker containerization, Kubernetes deployment, and application logging.

**Disclaimer:** This project is intended for educational purposes only. Its predictions are not medical diagnoses and must not be used for clinical decisions.

## 2. Project Architecture

```text
Cleveland Heart Disease Dataset
              |
              v
       Data Acquisition
              |
              v
 Data Cleaning and Preprocessing
              |
              v
   Exploratory Data Analysis
              |
              v
       Model Training
    (Logistic Regression
       and Random Forest)
              |
              v
    Cross-Validation and
      Model Evaluation
              |
              v
    MLflow Experiment Tracking
              |
              v
       Saved ML Pipeline
              |
              v
       FastAPI Prediction API
              |
              v
        Docker Container
              |
              v
    Kubernetes Deployment
              |
              v
       Health Checks and Logs
```

## 3. Project Structure

```text
heart-disease-mlops/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── __init__.py
│   ├── download_data.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── predict_test.py
│   └── api.py
├── tests/
│   ├── test_model.py
│   └── test_api.py
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

## 4. Technology Stack

- **Programming language:** Python 3.11
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn
- **Experiment tracking:** MLflow
- **Model serialization:** Joblib
- **API framework:** FastAPI
- **API server:** Uvicorn
- **Testing:** Pytest
- **Version control:** Git and GitHub
- **CI/CD:** GitHub Actions
- **Containerization:** Docker
- **Orchestration:** Kubernetes with Docker Desktop

## 5. Environment Setup

### Prerequisites

Install the following tools:

- Python 3.11
- Git
- Docker Desktop
- Kubernetes enabled in Docker Desktop

Clone the repository:

```powershell
git clone https://github.com/Dheeraj-a26/heart-disease-mlops.git
cd heart-disease-mlops
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Data Acquisition and Preparation

Download the dataset:

```powershell
python src/download_data.py
```

Clean and prepare the dataset:

```powershell
python src/prepare_data.py
```

The preprocessing process handles missing values and converts the original disease-severity field into a binary classification target.

- `0`: No disease detected in the dataset.
- `1`: Disease detected in the dataset.

The cleaned dataset contains 303 records and 14 columns: 13 input features and one target column.

The dataset contains missing values, particularly in the `ca` and `thal` features. Missing values are handled during preprocessing.

## 7. Exploratory Data Analysis

The Jupyter notebook in `notebooks/` contains exploratory data analysis.

The analysis includes:

- Dataset structure and data types.
- Missing-value inspection.
- Feature distribution histograms.
- Target class distribution.
- Correlation heatmap.
- Examination of relationships between numerical features.

The target distribution is approximately:

| Class | Records | Percentage |
|---|---:|---:|
| 0 | 164 | 54.1% |
| 1 | 139 | 45.9% |

These results describe the dataset and do not represent disease prevalence in the general population.

## 8. Feature Engineering and Model Development

The project uses numerical and categorical features.

### Numerical features

- Age
- Resting blood pressure
- Cholesterol
- Maximum heart rate achieved
- ST depression (`oldpeak`)

Numerical preprocessing uses median imputation and standard scaling.

### Categorical features

- Sex
- Chest pain type
- Fasting blood sugar
- Resting electrocardiographic results
- Exercise-induced angina
- Slope
- Number of major vessels (`ca`)
- Thalassemia-related category (`thal`)

Categorical preprocessing uses most-frequent imputation and one-hot encoding.

### Models

Two classification algorithms are evaluated:

1. Logistic Regression
2. Random Forest Classifier

The training pipeline uses a stratified train-test split and stratified five-fold cross-validation.

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- ROC-AUC
- Confusion matrix
- Classification report

Run the training pipeline:

```powershell
python src/train.py
```

The model comparison results are saved under `reports/`.

The selected model is saved to:

```text
models/heart_disease_model.joblib
```

### Evaluation results

The following results were obtained during the current experiment:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Cross-validation accuracy | 84.71% | 80.16% |
| Test accuracy | 88.52% | 88.52% |
| Test precision | 83.87% | 81.82% |
| Test recall | 92.86% | 96.43% |
| Test ROC-AUC | 96.65% | 94.64% |

Logistic Regression was selected based on its cross-validation performance and higher test ROC-AUC. Random Forest achieved higher recall on the held-out test split.

These results are specific to the current dataset split and should not be interpreted as evidence of clinical effectiveness.

## 9. MLflow Experiment Tracking

MLflow is used to track model training experiments.

The experiment records model parameters, evaluation metrics, and relevant artifacts.

Start the MLflow user interface:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open the following address in a browser:

http://127.0.0.1:5000

The experiment is named:

```text
heart-disease-classification
```

MLflow allows the results of Logistic Regression and Random Forest to be compared across runs.

## 10. Model Prediction

The saved model can be tested using:

```powershell
python src/predict_test.py
```

The script loads the serialized preprocessing and classification pipeline and produces a prediction for a sample input.

The preprocessing pipeline is stored together with the trained model so that the same transformations can be applied during inference.

## 11. FastAPI Prediction Service

The project provides a FastAPI service with the following endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | API information |
| `/health` | GET | Health and model-loaded status |
| `/predict` | POST | Generate a classification prediction |
| `/docs` | GET | Interactive API documentation |

Start the API locally:

```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Open the interactive documentation:

http://127.0.0.1:8000/docs

### Example prediction request

Send a POST request to `/predict` using the following JSON input:

```json
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
  "thal": 3
}
```

Example response:

```json
{
  "prediction": 0,
  "label": "disease_not_detected",
  "confidence": 0.6981,
  "class_probabilities": {
    "0": 0.6981,
    "1": 0.3019
  },
  "disclaimer": "Educational model output only; not a medical diagnosis."
}
```

The example response demonstrates the current model output for this sample input. Confidence represents the model's predicted class probability, not a clinically validated estimate of medical risk.

## 12. Automated Testing

The project uses Pytest to validate model and API behavior.

Run the test suite:

```powershell
python -m pytest tests/ -v
```

The tests cover model functionality and API behavior, including:

- API home endpoint.
- Health endpoint.
- Successful prediction requests.
- Invalid input handling.
- Model-related checks.

The current test suite contains 10 tests, which passed during local verification.

## 13. Continuous Integration and Delivery

GitHub Actions runs the configured CI workflow when changes are pushed to the repository or pull requests are opened against the `main` branch.

The workflow includes:

1. Checking out the repository.
2. Setting up Python.
3. Installing dependencies.
4. Downloading the dataset.
5. Preparing the dataset.
6. Training the models.
7. Running automated tests.

The workflow configuration is located at:

```text
.github/workflows/ci.yml
```

Successful GitHub Actions runs provide evidence that the configured pipeline completes in the CI environment.

## 14. Docker Containerization

Build the Docker image:

```powershell
docker build -t heart-disease-api:1.0 .
```

Run the container:

```powershell
docker run --rm -p 8000:8000 heart-disease-api:1.0
```

Check the health endpoint from another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

The API is packaged with its dependencies and saved model so it can run in a containerized environment.

## 15. Kubernetes Deployment

The project includes Kubernetes Deployment and Service manifests in the `k8s/` directory.

The local Kubernetes cluster is provided by Docker Desktop.

Check the current Kubernetes context:

```powershell
kubectl config current-context
```

Check the cluster nodes:

```powershell
kubectl get nodes
```

Apply the Deployment and Service:

```powershell
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Inspect the deployment:

```powershell
kubectl get deployments
kubectl get pods
kubectl get services
```

The application is deployed using the Docker image:

```text
heart-disease-api:1.0
```

The Deployment configures readiness and liveness probes using the `/health` endpoint and specifies container resource requests and limits.

### Access the API through Kubernetes

Start port forwarding:

```powershell
kubectl port-forward service/heart-disease-api-service 8001:8000
```

Keep this terminal running.

Open the API documentation:

http://127.0.0.1:8001/docs

Check the health endpoint from another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
```

Example health response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

The API was successfully tested through Kubernetes, including a prediction request that returned HTTP 200.

## 16. Monitoring and Logging

The API uses a health endpoint and Kubernetes readiness and liveness probes to check application availability.

Application logs are available through:

```powershell
kubectl logs deployment/heart-disease-api
```

The logs show application startup, health-check requests, and prediction requests.

The current implementation provides basic operational logging. A full Prometheus/Grafana metrics dashboard has not been implemented.

## 17. Reproducibility

To reproduce the workflow from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/download_data.py
python src/prepare_data.py
python src/train.py
python -m pytest tests/ -v
```

Docker and Kubernetes can then be used to package and deploy the API.

The exact model metrics may change if the dataset, dependencies, training configuration, or random seeds are changed.

## 18. Limitations

- The dataset contains only 303 records.
- The results depend on the selected training and test split.
- High test-set metrics do not establish clinical validity.
- Model confidence is not necessarily calibrated medical risk.
- The current deployment is intended for local educational demonstration.
- The implementation does not provide a production-grade medical service.
- Basic application logs are implemented, but a dedicated monitoring dashboard is not included.

## 19. Repository

GitHub repository:

https://github.com/Dheeraj-a26/heart-disease-mlops

## 20. Conclusion

This project demonstrates an end-to-end MLOps workflow for a heart disease classification model, covering data processing, model evaluation, experiment tracking, model serialization, automated testing, CI/CD, Docker containerization, Kubernetes deployment, and basic application logging.

The project illustrates how machine learning models can be packaged and deployed as API services with repeatable workflows and operational health checks.