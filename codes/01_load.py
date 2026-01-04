#%% 
import pandas as pd

learn = pd.read_csv(
    "../raw_data/learn_dataset.csv",
    dtype={"INSEE": str}
)

test = pd.read_csv(
    "../raw_data/test_dataset.csv",
    dtype={"INSEE": str}
)


print("Learn shape:", learn.shape)
print("Test shape :", test.shape)

print("uid unique (learn/test):", learn["uid"].is_unique, test["uid"].is_unique)
print("target in learn/test   :", "target" in learn.columns, "target" in test.columns)
print("INSEE dtype:", learn["INSEE"].dtype)

#%%

import os
from pathlib import Path
import pandas as pd

# ---- Rutas robustas ----
HERE = Path(__file__).resolve().parent      # .../codes
ROOT = HERE.parent                          # .../project root
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

# %%
