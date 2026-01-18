# Imports
import os
from pathlib import Path
import pandas as pd

# Directory paths
HERE = Path(__file__).resolve().parent      
ROOT = HERE.parent                          
DEFAULT_DATA_DIR = ROOT / "raw_data"
DATA_DIR = Path(os.environ.get("DATA_DIR", DEFAULT_DATA_DIR)).resolve()

# Helper functions
def must_exist(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"\n❌ Not found:\n  {path}\n\n"
            f"DATA_DIR actual:\n  {DATA_DIR}\n\n"
            f"Solutions:\n"
            f"1) Copy CSVs:\n   {DEFAULT_DATA_DIR}\n"
            f"2) Or execute with:\n"
            f"   export DATA_DIR='/ruta/a/tus/csv'\n"
        )
    return path

def load_core():
    learn = pd.read_csv(
        must_exist(DATA_DIR / "learn_dataset.csv"),
        dtype={"INSEE": "string"}
    )
    test = pd.read_csv(
        must_exist(DATA_DIR / "test_dataset.csv"),
        dtype={"INSEE": "string"}
    )
    return learn, test

def sanity_checks(learn: pd.DataFrame, test: pd.DataFrame):
    print("ROOT      :", ROOT)
    print("DATA_DIR  :", DATA_DIR)
    print("\nShapes")
    print("  learn:", learn.shape)
    print("  test :", test.shape)

    print("\nUID uniqueness")
    print("  learn:", learn["uid"].is_unique)
    print("  test :", test["uid"].is_unique)

    print("\nTarget presence")
    print("  learn has target:", "target" in learn.columns)
    print("  test  has target:", "target" in test.columns)

    if "INSEE" in learn.columns:
        print("\nINSEE dtype:", learn["INSEE"].dtype)

def main():
    learn, test = load_core()
    sanity_checks(learn, test)

if __name__ == "__main__":
    main()
