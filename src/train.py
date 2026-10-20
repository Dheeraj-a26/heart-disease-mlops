from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "heart_disease_clean.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

EXPERIMENT_NAME = "heart-disease-classification"

NUMERICAL_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]


def build_preprocessor() -> ColumnTransformer:
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        ),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {DATA_PATH}. "
            "Run 'python src/prepare_data.py' first."
        )

    df = pd.read_csv(DATA_PATH)

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scoring = [
        "accuracy",
        "precision",
        "recall",
        "roc_auc",
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
    )

    mlflow.set_experiment(EXPERIMENT_NAME)
    results = []
    trained_pipelines = {}

    for name, model in build_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", model),
            ]
        )

        with mlflow.start_run(run_name=name):
            # Record the model's hyperparameters.
            mlflow.log_params(model.get_params())

            # Record the model type and evaluation setup.
            mlflow.log_param("model_name", name)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("test_size", 0.20)
            mlflow.log_param("random_state", 42)

            # Evaluate using cross-validation on the training set.
            cv_results = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
            )

            pipeline.fit(X_train, y_train)
            trained_pipelines[name] = pipeline

            predictions = pipeline.predict(X_test)
            probabilities = pipeline.predict_proba(X_test)[:, 1]

            result = {
                "model": name,
                "cv_accuracy_mean": cv_results["test_accuracy"].mean(),
                "cv_precision_mean": cv_results["test_precision"].mean(),
                "cv_recall_mean": cv_results["test_recall"].mean(),
                "cv_roc_auc_mean": cv_results["test_roc_auc"].mean(),
                "test_accuracy": accuracy_score(y_test, predictions),
                "test_precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "test_recall": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "test_roc_auc": roc_auc_score(y_test, probabilities),
            }

            results.append(result)

            # Record all evaluation metrics.
            mlflow.log_metrics(
                {
                    metric: float(value)
                    for metric, value in result.items()
                    if metric != "model"
                }
            )

            print("\n" + "=" * 65)
            print(f"MODEL: {name}")
            print("=" * 65)

            for metric, value in result.items():
                if metric != "model":
                    print(f"{metric}: {value:.4f}")

            cm = confusion_matrix(y_test, predictions)

            print("\nConfusion matrix:")
            print(cm)

            print("\nClassification report:")
            print(
                classification_report(
                    y_test,
                    predictions,
                    target_names=["No disease", "Disease present"],
                    zero_division=0,
                )
            )

            # Save the confusion matrix as a run artifact.
            confusion_path = REPORTS_DIR / (
                name.lower().replace(" ", "_") + "_confusion_matrix.txt"
            )

            confusion_path.write_text(
                str(cm),
                encoding="utf-8",
            )

            mlflow.log_artifact(str(confusion_path))

            # Log the fitted pipeline, including preprocessing.
            mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            serialization_format="cloudpickle",
            )

    # Save the comparison table after both models finish.
    results_df = pd.DataFrame(results)

    results_path = REPORTS_DIR / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    print("\nMODEL COMPARISON")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {results_path}")

    # Save the comparison CSV as an artifact in a separate MLflow run.
    with mlflow.start_run(run_name="Model Comparison Summary"):
        mlflow.log_artifact(str(results_path))

        mlflow.log_param("models_compared", len(results_df))
        mlflow.log_param("selection_basis", "Cross-validation metrics")

    # Preserve our current provisional model selection.
    selected_model = trained_pipelines["Logistic Regression"]

    model_path = MODELS_DIR / "heart_disease_model.joblib"
    joblib.dump(selected_model, model_path)

    print(f"\nSelected model saved to: {model_path}")
    print(f"MLflow tracking database: {MLFLOW_DB_PATH}")


if __name__ == "__main__":
    main()