from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "heart_disease_raw.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "heart_disease_clean.csv"

COLUMNS = [
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
    "num",
]


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the Cleveland dataset and interpret '?' as missing."""
    return pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        na_values="?",
    )


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Create a binary target and validate the dataset."""
    data = df.copy()

    # The original target must be present.
    if "num" not in data.columns:
        raise ValueError("The dataset must contain the original 'num' target.")

    # The target must not contain missing values.
    if data["num"].isna().any():
        raise ValueError("The original target contains missing values.")

    # Convert the original 0-4 target into binary classification.
    data["target"] = (data["num"] > 0).astype(int)

    # Drop the original target to prevent accidental target leakage.
    data = data.drop(columns=["num"])

    return data


def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DATA_PATH}. "
            "Run 'python src/download_data.py' first."
        )

    raw_df = load_data()
    clean_df = prepare_data(raw_df)

    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(CLEAN_DATA_PATH, index=False)

    print(f"Raw dataset shape: {raw_df.shape}")
    print(f"Prepared dataset shape: {clean_df.shape}")
    print(f"Missing values before imputation: {clean_df.isna().sum().sum()}")
    print(f"Duplicate rows: {clean_df.duplicated().sum()}")
    print("\nBinary target distribution:")
    print(clean_df["target"].value_counts().sort_index())
    print(f"\nSaved prepared dataset to: {CLEAN_DATA_PATH}")


if __name__ == "__main__":
    main()
