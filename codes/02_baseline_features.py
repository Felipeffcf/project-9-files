# Baseline feature engineering script
# Imports
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Directory paths
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_DATA_DIR = ROOT / "raw_data"
DATA_DIR = Path(os.environ.get("DATA_DIR", DEFAULT_DATA_DIR)).resolve()

CTS_DIR = ROOT / "artifacts"
CTS_DIR.mkdir(exist_ok=True)

# Helper functions
def must_exist(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n  {path}\n\n"
            f"Current DATA_DIR:\n  {DATA_DIR}\n\n"
            f"Solutions:\n"
            f"1) Put the CSV files into {DEFAULT_DATA_DIR}\n"
            f"2) Or define DATA_DIR, e.g.:\n"
            f"   export DATA_DIR='/path/to/your/csvs'\n"
        )
    return path

def read_csv(name: str, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(must_exist(DATA_DIR / name), **kwargs)
    if "INSEE" in df.columns:
        df["INSEE"] = df["INSEE"].astype("string")
    return df

def left_join(base: pd.DataFrame, other: pd.DataFrame, on: str) -> pd.DataFrame:
    if other is None or other.empty:
        return base
    return base.merge(other, how="left", on=on)

def find_col(df: pd.DataFrame, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def load_job_description_mapping(data_dir: Path):
    """
    Loads mapping:
      JOB_DESCRIPTION code -> n2 code (from code_JOB_DESCRIPTION_map.csv)
      n2 code -> label (from code_JOB_DESCRIPTION_n2.csv, if label column exists)
    Returns (jd_to_n2, n2_to_label) or (None, None) if files/columns not found.
    """
    # Load mapping files
    map_path = data_dir / "code_JOB_DESCRIPTION_map.csv"
    n2_path = data_dir / "code_JOB_DESCRIPTION_n2.csv"

    if not map_path.exists() or not n2_path.exists():
        return None, None

    # Read mapping files
    m = pd.read_csv(map_path)
    n2 = pd.read_csv(n2_path)

    jd_col = find_col(m, ["JOB_DESCRIPTION", "job_description", "code", "pcs", "PCS"])
    n2_col = find_col(m, ["n2", "N2", "JOB_DESCRIPTION_n2", "code_n2"])

    n2_code_col = find_col(n2, ["n2", "N2", "code", "CODE"])
    n2_label_col = find_col(n2, ["label", "LABEL", "name", "NAME", "JOB_DESCRIPTION_n2"])

    if jd_col is None or n2_col is None or n2_code_col is None:
        return None, None

    jd_to_n2 = dict(zip(m[jd_col].astype(str), m[n2_col].astype(str)))

    if n2_label_col is not None:
        n2_to_label = dict(zip(n2[n2_code_col].astype(str), n2[n2_label_col].astype(str)))
    else:
        n2_to_label = {}

    return jd_to_n2, n2_to_label

def apply_job_mapping(df: pd.DataFrame, col: str, jd_to_n2: dict, n2_to_label: dict, out_col: str):
    """
    Adds out_col as mapped n2 category (label if available, otherwise code).
    Keeps None/NaN if original is missing or mapping not found.
    """
    s = df[col].astype("string")
    n2_code = s.map(lambda x: jd_to_n2.get(str(x), None) if pd.notna(x) else None)

    if n2_to_label:
        df[out_col] = n2_code.map(lambda x: n2_to_label.get(str(x), str(x)) if pd.notna(x) else None)
    else:
        df[out_col] = n2_code

    return df

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

    # prevent INSEE duplicates from creating duplication on merge
    if "INSEE" in geo.columns:
        geo = geo.drop_duplicates(subset=["INSEE"])

    return geo

# Job aggregation (from multiple rows to one row per uid)
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

    numeric_cols = ["stipend", "WORKING_HOURS"]

    categorical_cols = [
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

    for c in numeric_cols:
        if c in job.columns:
            agg_map[c] = "mean"

    for c in categorical_cols:
        if c in job.columns:
            agg_map[c] = mode_or_first

    if not agg_map:
        return job[["uid"]].drop_duplicates()

    return job.groupby("uid", as_index=False).agg(agg_map)

def aggregate_retired_last_job(retired_jobs: pd.DataFrame) -> pd.DataFrame:
    if retired_jobs is None or retired_jobs.empty:
        return pd.DataFrame({"uid": []})

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

    if df["uid"].is_unique:
        return df

    agg = {"uid": "first"}
    for col in keep:
        if col != "uid":
            agg[col] = mode_or_first
    return df.groupby("uid", as_index=False).agg(agg)

# Build Master Dataset (one row per person)
def build_master(split: str, geo: pd.DataFrame) -> pd.DataFrame:
    assert split in ["learn", "test"], "split must be 'learn' or 'test'"

    core = read_csv(f"{split}_dataset.csv")

    emp = read_csv(f"{split}_dataset_Emp_type.csv")
    core = left_join(core, emp, on="uid")
    core["has_emp_type"] = core["Emp_type"].notna().astype(int) if "Emp_type" in core.columns else 0

    job = read_csv(f"{split}_dataset_job.csv")
    job_agg = aggregate_job(job)
    core = left_join(core, job_agg, on="uid")

    if "stipend" in core.columns:
        core["is_employee"] = core["stipend"].notna().astype(int)
    elif "JOB_DESCRIPTION" in core.columns:
        core["is_employee"] = core["JOB_DESCRIPTION"].notna().astype(int)
    else:
        core["is_employee"] = 0

    retired_former = read_csv(f"{split}_dataset_retired_former.csv")
    core = left_join(core, retired_former, on="uid")
    core["is_retired"] = core["retirement_age"].notna().astype(int) if "retirement_age" in core.columns else 0

    retired_jobs_raw = read_csv(f"{split}_dataset_retired_jobs.csv")
    retired_jobs = aggregate_retired_last_job(retired_jobs_raw)
    core = left_join(core, retired_jobs, on="uid")

    pension = read_csv(f"{split}_dataset_retired_pension.csv")
    core = left_join(core, pension, on="uid")
    core["has_pension"] = core["pension_amount"].notna().astype(int) if "pension_amount" in core.columns else 0

    sports = read_csv(f"{split}_dataset_sport.csv")
    core = left_join(core, sports, on="uid")
    if "SPORTS" in core.columns:
        core["has_sports"] = core["SPORTS"].notna().astype(int)
        core["SPORTS"] = core["SPORTS"].fillna("none")
    else:
        core["has_sports"] = 0

    if "INSEE" in core.columns:
        core = left_join(core, geo, on="INSEE")

    if core["uid"].duplicated().any():
        core = core.drop_duplicates(subset=["uid"], keep="first")

    return core

def quick_checks(df: pd.DataFrame, name: str):
    print(f"\n== {name} ==")
    print("Shape:", df.shape)
    print("uid unique:", df["uid"].is_unique)

    check_cols = [
        "Emp_type",
        "stipend",
        "WORKING_HOURS",
        "JOB_DEP",
        "Occupational_status",
        "retirement_age",
        "pension_amount",
        "SPORTS",
        "population",
        "log_population",
        "JOB_DESCRIPTION_n2",
        "Previous_JOB_DESCRIPTION_n2",
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

    # Apply JOB_DESCRIPTION simplification (n2 mapping)
    jd_to_n2, n2_to_label = load_job_description_mapping(DATA_DIR)

    if jd_to_n2 is not None:
        if "JOB_DESCRIPTION" in learn_master.columns:
            learn_master = apply_job_mapping(learn_master, "JOB_DESCRIPTION", jd_to_n2, n2_to_label, "JOB_DESCRIPTION_n2")
            test_master = apply_job_mapping(test_master, "JOB_DESCRIPTION", jd_to_n2, n2_to_label, "JOB_DESCRIPTION_n2")

        if "Previous_JOB_DESCRIPTION" in learn_master.columns:
            learn_master = apply_job_mapping(
                learn_master, "Previous_JOB_DESCRIPTION", jd_to_n2, n2_to_label, "Previous_JOB_DESCRIPTION_n2"
            )
            test_master = apply_job_mapping(
                test_master, "Previous_JOB_DESCRIPTION", jd_to_n2, n2_to_label, "Previous_JOB_DESCRIPTION_n2"
            )

        # Drop raw high-cardinality columns to avoid huge OneHot matrices
        drop_raw = [c for c in ["JOB_DESCRIPTION", "Previous_JOB_DESCRIPTION"] if c in learn_master.columns]
        learn_master = learn_master.drop(columns=drop_raw, errors="ignore")
        test_master = test_master.drop(columns=drop_raw, errors="ignore")

        print("✅ JOB_DESCRIPTION mapped to n2 and raw columns dropped.")
    else:
        print("⚠️ JOB_DESCRIPTION mapping files not found; keeping raw JOB_DESCRIPTION codes.")

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
