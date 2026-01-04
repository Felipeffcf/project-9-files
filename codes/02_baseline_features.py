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
    df = pd.read_csv(
        must_exist(DATA_DIR / name),
        **kwargs)
    if "INSEE" in df.columns:
        df["INSEE"] = df["INSEE"].astype("string")
    return df

def left_join(base: pd.DataFrame, other: pd.DataFrame, on: str) -> pd.DataFrame:
    if other is None or other.empty:
        return base
    return base.merge(other, how="left", on=on)

# %%
# Geography table (INSEE --> region, population, coords)

def build_geo() -> pd.DataFrame:
    adm = read_csv("city_adm.csv")
    pop = read_csv("city_pop.csv")
    loc = read_csv("city_loc.csv")
    
    geo = (
        adm
        .merge(pop, on="INSEE", how="left")
        .merge(loc, on="INSEE", how="left")
    )
    
# Standardise population column

    if "POPULATION" in geo.columns:
        geo = geo.rename(columns={"POPULATION": "population"})
        
# Log-transformation of population (to reduce skewness)
    if "population" in geo.columns:
        geo["log_population"] = np.log1p(geo["population"])
    return geo

# %%
# Job aggregation (multiple rows --> one row per uid)

def mode_or_first(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return np.nan
    m = s.mode()
    if len(m) > 0:
        return m.iloc[0]
    return s.iloc[0]

def aggregate_job(job: pd.DataFrame) -> pd.DataFrame:
    if job is None or job.empty:
        return pd.DataFrame({"uid": []})
    
    numeric_cols = ["stipend", "WWORKING_HOURS"]
    caregorical_cols = [
        "activity_sector",
        "type_of_contract",
        "Occupational_status",
        "job_condition",
        "employer_type",
        "employee_count",
        "JOB_DEP",
        "JOB_DESCRIPTION",
    ]
    
    agg_map = {}

# Numeric: average accross multiple rows per uid
    for c in numeric_cols:
        if c in job.columns:
            agg_map[c] = "mean"

# Categorical: mode (most frequent) or first if no mode
        if c in job.columns:
            agg_map[c] = mode_or_first

# If for some reason no expected cols are present, return unique uids
    if not agg_map:
        return job[["uid"]].drop_duplicates()

    return job.groupby("uid", as_index=False).agg(agg_map)

# %% 

def aggregate_retired_last_job(retired_jobs: pd.DataFrame) -> pd.DataFrame:
    if retired_jobs is None or retired_jobs.empty:
        return pd.DataFrame({"uid": []})

# We keep only the columns that exist
    keep = ["uid"]
    candidates = [
        "Previous_JOB_DESCRIPTION",
        "PREVIOUS_DEP",
        "JOB_DEP",
        "JOB_DESCRIPTION",
    ]
    
    for col in candidates:
        if col in retired_jobs.columns:
            keep.append(col)
    df = retired_jobs[keep].copy()

# If already one row per uid, return as is
    if df["uid"].is_unique:
        return df
    
# otherwise, aggregate deterministically
    agg = {"uid": "first"}
    for col in keep:
        if col != "uid":
            agg[col] = mode_or_first
    return df.groupby("uid", as_index=False).agg(agg)
# %%
# Build Master Dataset (one row per person)

def build_master(split: str, geo: pd.DataFrame) -> pd.DataFrame:
    assert split in ["learn", "test"],"spilt must be 'learn' or 'test'"
    
# --- Core persons table ---
    core = read_csv(f"{split}_dataset.csv")

# --- Employment type ---
    emp = read_csv(f"{split}_dataset_Emp_type.csv")
    core = left_join(core, emp, on="uid")
    core["has_emp_type"] = core["Emp_type"].notna().astype(int) if "Emp_type" in core.columns else 0

# --- Job tables (employees only)---
    job = read_csv(f"{split}_dataset_job.csv") 
    job_agg = aggregate_job(job)
    core = left_join(core, job_agg, on="uid")

# Employee indicator (has at least one job record)
    if "stipend" in core.columns:
        core["is_employee"] = core["stipend"].notna().astype(int)
    elif "JOB_DESCRIPTION" in core.columns:
        core["is_employee"] = core["JOB_DESCRIPTION"].notna().astype(int)
    else:
        core["is_employee"] = 0

# --- Retired former & jobs tables ---
    retired_former = read_csv(f"{split}_dataset_retired_former.csv")
    core = left_join(core, retired_former,on="uid")
    core["is_retired"] = core["retirement_age"].notna().astype(int) if "retirement_age" in core.columns else 0

    retired_jobs_raw = read_csv(f"{split}_dataset_retired_jobs.csv")
    retired_jobs = aggregate_retired_last_job(retired_jobs_raw)
    core = left_join(core, retired_jobs, on="uid")

# --- Pension ---
    pension = read_csv(f"{split}_dataset_retired_pension.csv")
    core = left_join(core, pension, on="uid")
    core["has_pension"] = core["pension_amount"].notna().astype(int) if "pension_amount" in core.columns else 0

# --- Sports ---
    sports = read_csv(f"{split}_dataset_sport.csv")
    core = left_join(core, sports, on="uid")
    if "SPORTS" in core.columns:
        core["has_sports"] = core["SPORTS"].notna().astype(int)
        core["SPORTS"] = core["SPORTS"].fillna("none")
    else:
        core["has_sports"] = 0

# --- Geography ---
    if "INSEE" in core.columns:
        core = left_join(core, geo, on="INSEE")

# --- Final safety: ensure one row per uid ---
    if core["uid"].duplicated().any():
        core = core.drop_duplicates(subset=["uid"], keep="first")
    return core

# %%
def quick_checks(df: pd.DataFrame, name: str):
    print(f"\n== {name} ==")
    print("Shape:", df.shape)
    print("uid unique:", df["uid"].is_unique)
    
    check_cols =[
        "Emp_type",
        "stipend",
        "WORKING_HOURS",
        "JOB_DEP",
        "Occupational_status",
        "retirement_age",
        "pension_amount",
        "SPORTS",
        "population",
        "X",
        "Y",
    ]
    for col in check_cols:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            print(f"Column '{col}': missing values = {n_missing} ({n_missing / len(df):.2%})")

def main():
    print("Project ROOT:", ROOT)
    print("Using DATA_DIR:", DATA_DIR)
    print("CTS_DIR:", CTS_DIR)
    
    geo = build_geo()
    learn_master = build_master("learn", geo)
    test_master = build_master("test", geo)
    
    quick_checks(learn_master, "Learn Master Dataset")
    quick_checks(test_master, "Test Master Dataset")
    
    learn_path = CTS_DIR / "learn_master.csv"
    test_path = CTS_DIR / "test_master.csv"
    learn_master.to_csv(learn_path, index=False)
    test_master.to_csv(test_path, index=False)
    
    print("\n✅ Saved:")
    print("", learn_path)
    print("", test_path)
    
if __name__ == "__main__":
    main()

# %%
