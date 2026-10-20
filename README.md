# Heart Disease Classification --- MLOps Project

## 1. Project Overview

This project implements an end-to-end Machine Learning Operations
(MLOps) pipeline for binary heart disease classification using the
Cleveland Heart Disease dataset.

The workflow covers:

-   Dataset acquisition and preprocessing
-   Exploratory data analysis (EDA)
-   Feature preprocessing and model comparison
-   Cross-validation and held-out evaluation
-   MLflow experiment tracking
-   Model serialization and reproducibility
-   Automated tests and GitHub Actions CI
-   FastAPI prediction service
-   Docker containerization
-   Kubernetes deployment on a local Docker Desktop cluster
-   Cloud lab VM deployment using Docker
-   Basic API health checks, request counters, and prediction-latency
    metrics

> **Educational-use disclaimer:** This project is for learning and
> demonstration only. Its predictions are not medical diagnoses and must
> not be used for clinical decisions.

## 2. Architecture

``` text
Cleveland Heart Disease Dataset
              |
              v
       Data Acquisition
              |
              v
   Cleaning and Preparation
              |
              v
  EDA and Feature Inspection
              |
              v
  Preprocessing + Model Training
   (Logistic Regression and
        Random Forest)
              |
              v
 Cross-validation and Evaluation
              |
              v
       MLflow Tracking
              |
              v
 Saved preprocessing/model pipeline
              |
              v
        FastAPI service
       /health /predict /metrics
              |
       +------+------+
       |             |
       v             v
 Docker container  Kubernetes Deployment
 (lab VM)          (Docker Desktop local)
```

## 3. Repository Structure

``` text
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
│   └── 01_exploratory_data_analysis.ipynb
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
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

Generated datasets, model artifacts, and MLflow tracking files may be
created locally by the workflow and are not necessarily tracked by Git.

## 4. Technology Stack

-   **Language:** Python 3.11
-   **Data processing:** Pandas, NumPy
-   **Machine learning:** scikit-learn
-   **Experiment tracking:** MLflow
-   **Model serialization:** Joblib
-   **API framework:** FastAPI
-   **ASGI server:** Uvicorn
-   **Testing:** Pytest
-   **Version control:** Git and GitHub
-   **CI:** GitHub Actions
-   **Containerization:** Docker
-   **Orchestration:** Kubernetes on Docker Desktop
-   **Cloud/lab demonstration:** Docker container running on the lab VM

## 5. Environment Setup

### Prerequisites

Install Python 3.11, Git, and Docker. For the local Kubernetes steps,
enable Kubernetes in Docker Desktop and install `kubectl`.

Clone the repository:

``` powershell
git clone https://github.com/Dheeraj-a26/heart-disease-mlops.git
cd heart-disease-mlops
```

Create and activate a virtual environment in Windows PowerShell:

``` powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

``` powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Data Acquisition and Preparation

Download the dataset:

``` powershell
python src/download_data.py
```

Prepare the cleaned dataset:

``` powershell
python src/prepare_data.py
```

The source dataset contains 303 records and 14 original columns,
including the original disease-severity target. The preparation script
handles `?` values as missing data, creates a binary target from the
original target field, and saves the cleaned dataset.

Target encoding:

-   `0`: no disease indicated by the dataset label
-   `1`: disease indicated by the dataset label

The cleaned dataset contains 303 rows and 14 columns: 13 input features
and one binary target. Missing values occur in features including `ca`
and `thal`; the model pipeline imputes missing feature values during
training and inference.

## 7. Exploratory Data Analysis

The notebook `notebooks/01_exploratory_data_analysis.ipynb` contains the
EDA.

The analysis includes:

-   Dataset structure and data types
-   Missing-value inspection
-   Feature histograms
-   Target class distribution
-   Correlation heatmap
-   Relationships between selected numerical features

Target distribution in the cleaned dataset:

  Class     Records   Percentage
  ------- --------- ------------
  0             164        54.1%
  1             139        45.9%

These counts describe this dataset only; they do not represent disease
prevalence in the general population.

## 8. Feature Engineering and Model Development

The preprocessing pipeline separates numerical and categorical features.

### Numerical features

-   `age`
-   `trestbps`
-   `chol`
-   `thalach`
-   `oldpeak`

Numerical preprocessing uses median imputation and standard scaling.

### Categorical features

-   `sex`
-   `cp`
-   `fbs`
-   `restecg`
-   `exang`
-   `slope`
-   `ca`
-   `thal`

Categorical preprocessing uses most-frequent imputation and one-hot
encoding.

### Models and evaluation

Two classifiers are trained and compared:

1.  Logistic Regression
2.  Random Forest Classifier

The workflow uses a stratified train/test split and stratified five-fold
cross-validation. Evaluation includes accuracy, precision, recall,
ROC-AUC, confusion matrices, and classification reports.

Run training:

``` powershell
python src/train.py
```

The model comparison table is saved under `reports/`, and the selected
model pipeline is saved to:

``` text
models/heart_disease_model.joblib
```

### Recorded evaluation results

The following values are from the recorded experiment and depend on the
current dataset split and configuration.

  Metric                        Logistic Regression   Random Forest
  --------------------------- --------------------- ---------------
  Cross-validation accuracy                  84.71%          80.16%
  Test accuracy                              88.52%          88.52%
  Test precision                             83.87%          81.82%
  Test recall                                92.86%          96.43%
  Test ROC-AUC                               96.65%          94.64%

Logistic Regression was selected based on its cross-validation
performance and higher test ROC-AUC. Random Forest achieved higher
recall on this particular held-out split. These metrics do not establish
clinical effectiveness.

## 9. MLflow Experiment Tracking

MLflow records model parameters, metrics, and relevant artifacts for the
training experiments.

Start the MLflow UI from the project root:

``` powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open:

``` text
http://127.0.0.1:5000
```

The experiment is named:

``` text
heart-disease-classification
```

The UI can be used to compare the Logistic Regression and Random Forest
runs. Tracking files and the local SQLite database are generated during
execution and may be excluded from Git.

## 10. Run a Sample Model Prediction

Run the local prediction script:

``` powershell
python src/predict_test.py
```

The serialized artifact contains the preprocessing pipeline together
with the trained classifier, so inference applies the same feature
transformations used during training.

## 11. FastAPI Prediction Service

The API provides the following endpoints:

  Endpoint     Method   Purpose
  ------------ -------- -----------------------------------------------
  `/`          GET      API information
  `/health`    GET      Health status and model-loaded status
  `/predict`   POST     Predict the dataset class for an input record
  `/metrics`   GET      Basic request and prediction-latency metrics
  `/docs`      GET      Interactive API documentation

Start the API locally:

``` powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

In another terminal, check health:

``` powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Open interactive documentation:

``` text
http://127.0.0.1:8000/docs
```

### Example prediction input

Send a `POST` request to `/predict` with JSON containing all 13
features:

``` json
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

A recorded example response for this input was:

``` json
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

The output is a model classification based on the dataset's labels. The
returned class probability is not a clinically validated estimate of a
person's medical risk. The model requires all 13 input features; it does
not make predictions from age and sex alone.

## 12. Automated Testing

The project uses Pytest for model and API tests.

Run tests from the repository root:

``` powershell
python -m pytest tests/ -v
```

The tests cover model functionality and API behavior, including the home
endpoint, health endpoint, prediction requests, invalid inputs, and the
metrics endpoint.

The most recent recorded local test result was **11 tests passed**.
Re-run the suite after any code or dependency changes and update this
statement if the result changes.

## 13. Continuous Integration

The GitHub Actions workflow is defined in:

``` text
.github/workflows/ci.yml
```

The configured workflow performs these steps:

1.  Checks out the repository.
2.  Sets up Python.
3.  Installs dependencies.
4.  Downloads the dataset.
5.  Prepares the dataset.
6.  Trains the models.
7.  Runs automated tests.

The four workflow runs visible in GitHub Actions, including the latest
run for commit `f2e0b49` (`Add API monitoring metrics and tests`),
showed green success indicators at the time of verification. Check the
Actions tab for the latest status after future pushes.

Repository: https://github.com/Dheeraj-a26/heart-disease-mlops

## 14. Docker Containerization

Build the Docker image from the repository root:

``` powershell
docker build -t heart-disease-api:1.1 .
```

Run the container:

``` powershell
docker run --rm --name heart-disease-api-container -p 8000:8000 heart-disease-api:1.1
```

Check health from another terminal:

``` powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Open the API documentation at:

``` text
http://127.0.0.1:8000/docs
```

The Docker image packages the Python runtime, dependencies, API source,
and serialized model needed to serve predictions. The image does not
contain every repository file, such as notebooks and tests.

### Lab VM deployment verification

The image `heart-disease-api:1.1` was built and run on the lab VM. The
following commands were successfully tested on that VM:

``` bash
docker ps
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

A sample `POST /predict` request returned a prediction response. After
one successful prediction, the metrics endpoint reported:

-   `successful_predictions`: `1`
-   `failed_predictions`: `0`
-   `prediction_response_time_ms_last`: approximately `110.74`
-   `average_prediction_response_time_ms`: approximately `110.74`

These are observations from that test session, not fixed values. The
VM's local address `localhost:8000` is accessible from within the VM;
external access depends on the lab's networking and firewall
configuration.

To inspect application logs:

``` bash
docker logs --tail 50 heart-disease-api-container
```

## 15. Kubernetes Deployment (Docker Desktop)

The project includes Kubernetes Deployment and Service manifests in
`k8s/`. The Kubernetes demonstration was tested on a local Docker
Desktop cluster. This is separate from the Docker container running on
the lab VM.

Check the current context and nodes:

``` powershell
kubectl config current-context
kubectl get nodes
```

Apply the manifests:

``` powershell
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Inspect the deployment:

``` powershell
kubectl get deployments
kubectl get pods
kubectl get services
```

Confirm that the image referenced by `k8s/deployment.yaml` matches the
image tag available in the local Docker Desktop environment. If you
change the manifest, reapply it.

Forward the service port:

``` powershell
kubectl port-forward service/heart-disease-api-service 8001:8000
```

Keep that terminal running. Open:

``` text
http://127.0.0.1:8001/docs
```

Check health from another terminal:

``` powershell
Invoke-RestMethod http://127.0.0.1:8001/health
```

The Kubernetes deployment was previously verified with a running pod and
a successful prediction request. Re-run the commands and capture fresh
screenshots for the final submission.

## 16. Monitoring and Logging

The FastAPI application provides a `/health` endpoint and a `/metrics`
endpoint.

The `/metrics` response includes:

-   `total_requests`
-   `successful_predictions`
-   `failed_predictions`
-   `prediction_response_time_ms_total`
-   `prediction_response_time_ms_last`
-   `average_prediction_response_time_ms`

Example command:

``` bash
curl http://localhost:8000/metrics
```

The implementation uses in-memory counters and latency values. These
reset when the application process restarts and are not a replacement
for persistent, multi-instance production monitoring. A dedicated
Prometheus/Grafana dashboard has not been implemented.

Docker logs can be inspected using `docker logs`; Kubernetes logs can be
inspected using:

``` powershell
kubectl logs deployment/heart-disease-api
```

## 17. Reproducibility

From the repository root, the core workflow can be rerun using:

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/download_data.py
python src/prepare_data.py
python src/train.py
python -m pytest tests/ -v
```

Then run MLflow, the API, Docker, or Kubernetes as described above.

Metrics can vary if the dataset, dependencies, random seeds, or training
configuration change. Only load model artifacts from trusted sources
because serialized Python model files can pose a security risk if
untrusted.

## 18. Limitations

-   The dataset contains only 303 records.
-   Evaluation results depend on the selected train/test split and
    experimental setup.
-   High test-set metrics do not establish clinical validity.
-   Predicted class probabilities may not be calibrated medical risks.
-   The API requires all 13 model features.
-   Basic in-memory monitoring is implemented, but no dedicated
    Prometheus/Grafana dashboard is included.
-   The local Kubernetes deployment and lab VM Docker deployment are
    separate demonstrations.
-   This is an educational project, not a production-grade medical
    service.

## 19. Final Validation

Run these commands from the repository root before submission.

### Run the automated tests

```powershell
python -m pytest tests/ -v
```

### Verify the API

Start the API or Docker container, then run:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

Send the sample JSON request from Section 11 to `POST /predict`, then query `/metrics` again to verify that the successful-prediction counter and latency values update.

### Verify Docker

```bash
docker ps
docker logs --tail 50 heart-disease-api-container
```

### Verify Kubernetes (Docker Desktop)

```powershell
kubectl get deployments
kubectl get pods
kubectl get services
kubectl logs deployment/heart-disease-api
```

For the assignment report, describe the implemented workflow and include the actual results produced by these checks. Capture evidence only where the assignment submission instructions require it; use screenshots directly from the running application or terminal rather than fabricated examples.

## 20. Repository

GitHub repository:

https://github.com/Dheeraj-a26/heart-disease-mlops

## 21. Conclusion

This project demonstrates an end-to-end educational MLOps workflow for
heart disease classification: data acquisition, preprocessing, EDA,
model comparison, MLflow experiment tracking, model serialization,
automated testing, GitHub Actions CI, FastAPI serving, Docker
containerization, Kubernetes deployment, and basic API monitoring.

It illustrates how a trained machine learning pipeline can be packaged
and served reproducibly, while highlighting the limitations of small
datasets and the distinction between an educational classification
output and a medical diagnosis.
