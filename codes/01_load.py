#%%
import os
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent      
ROOT = HERE.parent                          
DEFAULT_DATA_DIR = ROOT / "raw_data"
DATA_DIR = Path(os.environ.get("DATA_DIR", DEFAULT_DATA_DIR)).resolve()

def must_exist(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"\n❌ No encuentro:\n  {path}\n\n"
            f"DATA_DIR actual:\n  {DATA_DIR}\n\n"
            f"Soluciones:\n"
            f"1) Copia los CSV del profesor a:\n   {DEFAULT_DATA_DIR}\n"
            f"2) O ejecuta con:\n"
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

# %% FF
"""
01_load.py
- Carga learn_dataset.csv y test_dataset.csv desde raw_data/
- Hace checks básicos y prints útiles (shapes, columnas clave)
"""

import os
from pathlib import Path
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_DATA_DIR = ROOT / "raw_data"
DATA_DIR = Path(os.environ.get("DATA_DIR", DEFAULT_DATA_DIR)).resolve()


def must_exist(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(
            f"File not found:\n  {p}\n\n"
            f"Current DATA_DIR:\n  {DATA_DIR}\n\n"
            f"Fix:\n"
            f" - Put CSVs in: {DEFAULT_DATA_DIR}\n"
            f" - or set DATA_DIR env var to your raw_data folder."
        )
    return p


def load_core() -> tuple[pd.DataFrame, pd.DataFrame]:
    learn_path = must_exist(DATA_DIR / "learn_dataset.csv")
    test_path = must_exist(DATA_DIR / "test_dataset.csv")

    learn = pd.read_csv(learn_path, dtype={"INSEE": str})
    test = pd.read_csv(test_path, dtype={"INSEE": str})
    return learn, test


def sanity_checks(learn: pd.DataFrame, test: pd.DataFrame) -> None:
    print("DATA_DIR:", DATA_DIR)
    print("Learn shape:", learn.shape)
    print("Test  shape:", test.shape)

    for name, df in [("learn", learn), ("test", test)]:
        if "uid" not in df.columns:
            raise ValueError(f"{name} is missing column 'uid'.")
        if df["uid"].isna().any():
            raise ValueError(f"{name} has NA in 'uid'.")
        if df["uid"].duplicated().any():
            raise ValueError(f"{name} has duplicated 'uid' values.")

    print("Columns (learn) sample:", list(learn.columns)[:15], "...")
    print("Has target in learn:", "target" in learn.columns)
    print("Has target in test :", "target" in test.columns)
    if "INSEE" in learn.columns:
        print("INSEE dtype:", learn["INSEE"].dtype)


def main() -> None:
    learn, test = load_core()
    sanity_checks(learn, test)


if __name__ == "__main__":
    main()

# %%
