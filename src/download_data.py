from pathlib import Path
from urllib.request import urlretrieve

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
OUTPUT_FILE = DATA_DIR / "heart_disease_raw.csv"

def main():
    if OUTPUT_FILE.exists():
        print(f"Dataset already exists: {OUTPUT_FILE}")
        print("Delete the file if you want to download it again.")
        return

    print("Downloading the Cleveland Heart Disease dataset...")
    urlretrieve(URL, OUTPUT_FILE)
    print(f"Download complete: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
