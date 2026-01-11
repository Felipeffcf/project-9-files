

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
"""
02_baseline_features.py
- Constructs learn_master.csv & test_master.csv in artifacts/
- Put uid features extra together (job, emp_type, retired_*, sport)
- Put geographic features together INSEE-level (city_adm, city_pop, city_loc, regions, departments)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_DATA_DIR = ROOT / "raw_data"
DATA_DIR = Path(os.environ.get("DATA_DIR", DEFAULT_DATA_DIR)).resolve()

ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


# ---------- Helpers ----------
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


def read_csv(name: str, **kwargs) -> pd.DataFrame:
    path = must_exist(DATA_DIR / name)
    return pd.read_csv(path, **kwargs)


def assert_uid_unique(df: pd.DataFrame, name: str) -> None:
    if "uid" not in df.columns:
        raise ValueError(f"{name} missing 'uid'.")
    if df["uid"].duplicated().any():
        raise ValueError(f"{name} has duplicated uid (should be 1 row per uid).")


def safe_mode(s: pd.Series):
    s = s.dropna()
    if s.empty:
        return np.nan
    m = s.mode()
    return m.iloc[0] if not m.empty else s.iloc[0]


def aggregate_uid(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Reduce a potentially 1-to-many table to 1 row per uid.
    - numeric cols -> mean
    - categorical/object cols -> mode
    """
    if df.empty:
        return df

    if "uid" not in df.columns:
        raise ValueError(f"{name}: missing uid column.")

    # drop duplicate column names just in case
    df = df.loc[:, ~df.columns.duplicated()]

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "uid"]
    other_cols = [c for c in df.columns if c not in numeric_cols and c != "uid"]

    agg = {}
    for c in numeric_cols:
        agg[c] = "mean"
    for c in other_cols:
        agg[c] = safe_mode

    out = df.groupby("uid", as_index=False).agg(agg)
    assert_uid_unique(out, name + "_agg")
    return out


def left_join_one_to_one(base: pd.DataFrame, add: pd.DataFrame, on: str, name: str) -> pd.DataFrame:
    """
    Left join with safety: 'add' must be unique on key.
    """
    if add.empty:
        return base
    if on not in add.columns:
        raise ValueError(f"{name}: missing join key '{on}' in add dataframe.")
    if add[on].duplicated().any():
        raise ValueError(f"{name}: add dataframe is not unique on '{on}'.")
    return base.merge(add, on=on, how="left")


# ---------- Geo ----------
def build_geo() -> pd.DataFrame:
    """
    Build geo table keyed by INSEE.
    Tries to merge: city_adm, city_loc, city_pop, departments, regions if present.
    """
    geo_parts = []

    # Each file should contain INSEE (or a similar city code). We handle a few common variants.
    def normalize_insee(df: pd.DataFrame) -> pd.DataFrame:
        # find a plausible INSEE key column
        for cand in ["INSEE", "insee", "insee_code", "code_insee", "CODGEO", "COD_GEO"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "INSEE"})
                break
        if "INSEE" not in df.columns:
            raise ValueError("Geo file missing INSEE-like key column.")
        df["INSEE"] = df["INSEE"].astype(str)
        return df

    # city_adm
    try:
        adm = normalize_insee(read_csv("city_adm.csv"))
        geo_parts.append(adm)
    except FileNotFoundError:
        pass

    # city_loc
    try:
        loc = normalize_insee(read_csv("city_loc.csv"))
        geo_parts.append(loc)
    except FileNotFoundError:
        pass

    # city_pop (create log_population if possible)
    try:
        pop = normalize_insee(read_csv("city_pop.csv"))
        # attempt to find population column
        pop_col = None
        for cand in ["population", "POPULATION", "pop", "POP"]:
            if cand in pop.columns:
                pop_col = cand
                break
        if pop_col:
            pop["log_population"] = np.log1p(pd.to_numeric(pop[pop_col], errors="coerce"))
        geo_parts.append(pop)
    except FileNotFoundError:
        pass

    # departments
    try:
        dep = read_csv("departments.csv")
        geo_parts.append(dep)
    except FileNotFoundError:
        pass

    # regions
    try:
        reg = read_csv("regions.csv")
        geo_parts.append(reg)
    except FileNotFoundError:
        pass

    if not geo_parts:
        # no geo files found
        return pd.DataFrame(columns=["INSEE"])

    # merge sequentially on INSEE where possible
    geo = geo_parts[0]
    if "INSEE" in geo.columns:
        geo["INSEE"] = geo["INSEE"].astype(str)

    for part in geo_parts[1:]:
        if "INSEE" in geo.columns and "INSEE" in part.columns:
            geo = geo.merge(part, on="INSEE", how="left", suffixes=("", "_dup"))
            # drop any duplicate suffix columns if they appear
            dup_cols = [c for c in geo.columns if c.endswith("_dup")]
            if dup_cols:
                geo = geo.drop(columns=dup_cols)
        else:
            # if part has no INSEE, skip (still better than crashing)
            continue

    # ensure unique INSEE if present
    if "INSEE" in geo.columns and geo["INSEE"].duplicated().any():
        geo = geo.drop_duplicates(subset=["INSEE"])

    return geo


# ---------- Master build ----------
def build_master(split: str, geo: pd.DataFrame) -> pd.DataFrame:
    """
    split: 'learn' or 'test'
    """
    core = read_csv(f"{split}_dataset.csv", dtype={"INSEE": str})
    if "uid" not in core.columns:
        raise ValueError(f"{split}_dataset.csv missing 'uid'.")
    if core["uid"].duplicated().any():
        raise ValueError(f"{split}_dataset.csv has duplicate uids.")

    # Emp type
    emp = read_csv(f"{split}_dataset_Emp_type.csv")
    emp = aggregate_uid(emp, f"{split}_dataset_Emp_type")
    core = left_join_one_to_one(core, emp, on="uid", name="emp_type")

    # Jobs
    job = read_csv(f"{split}_dataset_job.csv")
    job = aggregate_uid(job, f"{split}_dataset_job")
    core = left_join_one_to_one(core, job, on="uid", name="job")

    # Retired former
    rf = read_csv(f"{split}_dataset_retired_former.csv")
    rf = aggregate_uid(rf, f"{split}_dataset_retired_former")
    core = left_join_one_to_one(core, rf, on="uid", name="retired_former")

    # Retired jobs
    rj = read_csv(f"{split}_dataset_retired_jobs.csv")
    rj = aggregate_uid(rj, f"{split}_dataset_retired_jobs")
    core = left_join_one_to_one(core, rj, on="uid", name="retired_jobs")

    # Retired pension
    rp = read_csv(f"{split}_dataset_retired_pension.csv")
    rp = aggregate_uid(rp, f"{split}_dataset_retired_pension")
    core = left_join_one_to_one(core, rp, on="uid", name="retired_pension")

    # Sports
    sp = read_csv(f"{split}_dataset_sport.csv")
    sp = aggregate_uid(sp, f"{split}_dataset_sport")
    core = left_join_one_to_one(core, sp, on="uid", name="sport")

    # has_sports feature if SPORTS present
    if "SPORTS" in core.columns:
        core["has_sports"] = core["SPORTS"].notna().astype(int)
        core["SPORTS"] = core["SPORTS"].fillna("none")
    else:
        core["has_sports"] = 0

    # Geography on INSEE
    if "INSEE" in core.columns and "INSEE" in geo.columns:
        core["INSEE"] = core["INSEE"].astype(str)
        geo2 = geo.copy()
        geo2["INSEE"] = geo2["INSEE"].astype(str)
        geo2 = geo2.drop_duplicates(subset=["INSEE"])
        core = core.merge(geo2, on="INSEE", how="left")

    return core


def main() -> None:
    print("DATA_DIR:", DATA_DIR)
    geo = build_geo()

    learn_master = build_master("learn", geo)
    test_master = build_master("test", geo)

    # final checks
    if learn_master["uid"].duplicated().any():
        raise ValueError("learn_master has duplicated uid after merges.")
    if test_master["uid"].duplicated().any():
        raise ValueError("test_master has duplicated uid after merges.")

    learn_path = ARTIFACTS_DIR / "learn_master.csv"
    test_path = ARTIFACTS_DIR / "test_master.csv"

    learn_master.to_csv(learn_path, index=False)
    test_master.to_csv(test_path, index=False)

    print("\n✅ Saved:")
    print(" ", learn_path)
    print(" ", test_path)
    print("\nLearn master shape:", learn_master.shape)
    print("Test  master shape:", test_master.shape)


if __name__ == "__main__":
    main()
# %%
