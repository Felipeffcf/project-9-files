#%%
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def load_core():
    learn = pd.read_csv(
        r"C:\Users\felif\Desktop\Machine Learning\FINAL PROJECT\project-9-files\raw_data\learn_dataset.csv",
        dtype={"INSEE": str}
    )
    test = pd.read_csv(
        r"C:\Users\felif\Desktop\Machine Learning\FINAL PROJECT\project-9-files\raw_data\test_dataset.csv",
        dtype={"INSEE": str}
    )
    return learn, test

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Identify columns
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),  # fills NaN with mode
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor

if __name__ == "__main__":
    learn, test = load_core()
    X = learn.drop(columns=["target"])
    y = learn["target"]

    pre = make_preprocessor(X)
    Xt = pre.fit_transform(X)

    print("Preprocessed train matrix shape:", Xt.shape)
    print("y shape:", y.shape)

#%%
import os
from pathlib import Path
import pandas as pd
import numpy as np

# %%
HERE = Path(__file__).resolve().parent  
ROOT = HERE.parent
DEFAULT_DATA_DIR = ROOT / "raw_data"
DATA_DIR = Path(os.environ.get("DATA_DIR", DEFAULT_DATA_DIR)).resolve()

CTS_DIR = ROOT / "artifacts"
CTS_DIR.mkdir(exist_ok=True)

def must_exist(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n  {path}\n\n"
            f"Current DATA_DIR:\n  {DATA_DIR}\n\n"
            f"Solutiones:\n"
            f"1) Put the CSV files into {DEFAULT_DATA_DIR}\n"
            f"2) Or define DATA_DIR, e.g.:\n"
            f"   export DATA_DIR='/path/to/your/csvs'\n"
        )
    return path

def read_csv(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(
        must_exist(DATA_DIR / name),
        dtype={"INSEE": "string"},
        **kwargs
    )

def left_join(base: pd.DataFrame, other: pd.DataFrame, on: str) -> pd.DataFrame:
    if other is None or other.empty:
        return base
    return base.merge(other, how="left", on=on)

# %%
# Geography table (INSEE --> region, population, coords)

def build_geo() -> pd.DataFrame:
    geo = read_csv("geo_data.csv")
    geo = geo.rename(columns={
        "region_code": "region",
        "population_2017": "population",
        "latitude": "lat",
        "longitude": "lon",
    })
    geo = geo[["INSEE", "region", "population", "lat", "lon"]]
    return geo