#%%

import pandas as pd

LEARN = r"C:\Users\felif\Desktop\Machine Learning\FINAL PROJECT\project-9-files\raw_data\learn_dataset.csv"
TEST  = r"C:\Users\felif\Desktop\Machine Learning\FINAL PROJECT\project-9-files\raw_data\test_dataset.csv"

LEARN_SPORT = r"C:\Users\felif\Desktop\Machine Learning\FINAL PROJECT\project-9-files\raw_data\learn_dataset_sport.csv"
TEST_SPORT  = r"C:\Users\felif\Desktop\Machine Learning\FINAL PROJECT\project-9-files\raw_data\test_dataset_sport.csv"

OUT_LEARN = r"C:\Users\felif\Desktop\Machine Learning\FINAL PROJECT\project-9-files\raw_data\learn_plus_sport.csv"
OUT_TEST  = r"C:\Users\felif\Desktop\Machine Learning\FINAL PROJECT\project-9-files\raw_data\test_plus_sport.csv"

def add_sport_flag(core_path, sport_path, out_path):
    core = pd.read_csv(core_path, dtype={"INSEE": str})
    sport = pd.read_csv(sport_path)

    # People not listed are NOT members (per project statement)
    sport_uids = set(sport["uid"].unique())
    core["is_sport_member"] = core["uid"].isin(sport_uids).astype(int)

    core.to_csv(out_path, index=False)
    print("Saved:", out_path, "shape:", core.shape)

if __name__ == "__main__":
    add_sport_flag(LEARN, LEARN_SPORT, OUT_LEARN)
    add_sport_flag(TEST,  TEST_SPORT,  OUT_TEST)
#%%

import os 
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DEFAULT_DATA_DIR = ROOT / "raw_data"
DATA_DIR = Path(os.environ.get("DATA_DIR", str(DEFAULT_DATA_DIR))).resolve()

ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

LEARN_IN = Path(os.environ.get("LEARN_MASTER", ARTIFACTS_DIR / "learn_master.csv")).resolve()
TEST_IN  = Path(os.environ.get("TEST_MASTER",  ARTIFACTS_DIR / "test_master.csv")).resolve()

LEARN_SPORT = DATA_DIR / "learn_dataset_sport.csv"
TEST_SPORT  = DATA_DIR / "test_dataset_sport.csv"

LEARN_OUT = Path(os.environ.get("LEARN_MASTER_SPORT", ARTIFACTS_DIR / "learn_master_plus_sport.csv")).resolve()
TEST_OUT = Path(os.environ.get("TEST_MASTER_SPORT",  ARTIFACTS_DIR / "test_master_plus_sport.csv")).resolve()

def must_exist(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path

def add_sport_features(core_path: Path, sport_path: Path, out_path: Path) -> None:
    core = pd.read_csv(must_exist(core_path))
    sport = pd.read_csv(must_exist(sport_path))

    if "uid" not in core.columns: 
        raise ValueError(f" uid column missing in core dataset: {core_path}")
    if "uid" not in sport.columns:
        raise KeyError(f" uid column missing in sport dataset: {sport_path}")
    
    #People not listed are NOT members (per project statement)
    sport_uids = set(sport["uid"].dropna().unique())
    
    #Flag aligned with our pipeline naming
    core["has_sports"] = core["uid"].isin(sport_uids).astype(int)
    
    # If sprts column exists in sport table, merge it: otherwise fill NONE
    if "SPORTS" in sport.columns:
        sprt_small = sport[["uid", "SPORTS"]].drop_duplicates(subset=["uid"], keep="first")
        core = core.merge(sprt_small, on="uid", how="left")
        core["SPORTS"] = core["SPORTS"].fillna("NONE")
    else:
        core["SPORTS"] = core.get("SPORTS", pd.Series(["NONE"] * len(core))).fillna("NONE")

    core.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(" shape:", core.shape)
    print(" has_sports mean:", core["has_sports"].mean())

def main():
    print("ROOT :", ROOT)
    print("DATA_DIR :", DATA_DIR)
    print("LEARN_IN :", LEARN_IN)
    print("TEST_IN  :", TEST_IN)
    
    add_sport_features(LEARN_IN, LEARN_SPORT, LEARN_OUT)
    add_sport_features(TEST_IN, TEST_SPORT, TEST_OUT)

if __name__ == "__main__":
    main()
    
# %%