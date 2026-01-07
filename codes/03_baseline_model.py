#%%

import os
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# --- Paths (Option 2: absolute raw strings) ---
DATA_LEARN = r"C:\Users\felif\Desktop\Machine Learning\FINAL PROJECT\project-9-files\raw_data\learn_dataset.csv"

# --- Debug knobs ---
SUBSAMPLE_N = 15000        # set to None to use full data
N_JOBS = 4                 # reduce if your PC struggles; set -1 if you have lots of RAM/CPU
VERBOSE = 3                # print progress from GridSearch

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )

def main():
    print("CWD:", os.getcwd(), flush=True)
    print("Loading:", DATA_LEARN, flush=True)

    learn = pd.read_csv(DATA_LEARN, dtype={"INSEE": str})
    print("Loaded learn shape:", learn.shape, flush=True)

    X = learn.drop(columns=["target"])
    y_raw = learn["target"].astype(str)

    # Encode target labels -> 0..K-1
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print("Target classes:", list(le.classes_), flush=True)
    counts = dict(zip(*np.unique(y, return_counts=True)))
    print("Class counts:", counts, flush=True)

    # Subsample for fast debugging
    if SUBSAMPLE_N is not None and SUBSAMPLE_N < len(X):
        X = X.sample(n=SUBSAMPLE_N, random_state=42)
        y = y[X.index]
        print(f"Using subsample n={SUBSAMPLE_N}. X shape:", X.shape, flush=True)

    pre = make_preprocessor(X)

    clf = RandomForestClassifier(
        random_state=42,
        n_jobs=N_JOBS,
        class_weight="balanced_subsample",
    )

    pipe = Pipeline([
        ("pre", pre),
        ("rf", clf),
    ])

    # Small grid for a "smoke test" (fast)
    param_grid = {
        "rf__n_estimators": [200],
        "rf__max_depth": [None, 20],
        "rf__min_samples_leaf": [1, 5],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Starting GridSearchCV...", flush=True)
    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=N_JOBS,
        verbose=VERBOSE,
        error_score="raise",   # if something fails, show the real error immediately
    )

    gs.fit(X, y)

    print("\nBest CV f1_macro:", gs.best_score_, flush=True)
    print("Best params:", gs.best_params_, flush=True)

    # Holdout check
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    best_model = gs.best_estimator_
    best_model.fit(Xtr, ytr)
    pred = best_model.predict(Xte)

    print("\nHoldout accuracy:", accuracy_score(yte, pred), flush=True)
    print("Holdout f1_macro:", f1_score(yte, pred, average="macro"), flush=True)
    print("\nConfusion matrix:\n", confusion_matrix(yte, pred), flush=True)
    print("\nClassification report:\n", classification_report(yte, pred, target_names=le.classes_), flush=True)

if __name__ == "__main__":
    main()

#%%
import os
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_DATA_DIR = ROOT / "raw_data"
DATA_DIR = Path(os.environ.get("DATA_DIR", DEFAULT_DATA_DIR)).resolve()

ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

LEARN_MASTER = ARTIFACTS_DIR / "learn_master.csv"
TEST_MASTER = ARTIFACTS_DIR / "test_master.csv"

def must_exist(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n  {path}\n\n"
        )
    return path

def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(
        must_exist(path),
        **kwargs
    )

def _find_col(df: pd.DataFrame, candidates):
    cols = {c.lower() : c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def load_mapping_job_description(): 
    map_path = DATA_DIR / "code_JOB_DESCRIPTION_map.csv"
    n2_path = DATA_DIR / "code_JOB_DESCRIPTION_n2.csv"
    n1_path = DATA_DIR / "code_JOB_DESCRIPTION_n1.csv"
    
    if not map_path.exists() or not n2_path.exists():
        return None, None 

    m = pd.read_csv(map_path)
    n2 = pd.read_csv(n2_path)

# Infer Columns
    jd_col = _find_col(m, ["JOB_DESCRIPTION", "job_description", "code", "pcs", "PCS"])
    n2_col = _find_col(m, ["n2", "N2", "JOB_DESCRIPTION_n2", "code_n2"])
    
    n2_code_col = _find_col(n2, ["n2", "N2", "code", "CODE"])
    n2_label_col = _find_col(n2, ["label", "LABEL", "JOB_DESCRIPTION_n2", "name","NAME"])

    if jd_col is None or n2_col is None or n2_code_col is None :
        return None, None

# Build mapping
    jd_to_n2code = dict(
        zip(m[jd_col].astype(str), m[n2_col].astype(str))
    )

    if n2_label_col is not None:
        n2code_to_label = dict(
            zip(n2[n2_code_col].astype(str), n2[n2_label_col].astype(str))
        )
    else:
        n2code_to_label = {}
        
    return jd_to_n2code, n2code_to_label

# n1
    n1code_to_label = None
    if n1_path.exists():
        n1 = pd.read_csv(n1_path)
        n1_code_col = _find_col(n1, ["n1", "N1", "code", "CODE"])
        n1_label_col = _find_col(n1, ["label", "LABEL", "name","NAME"])
        if n1_code_col is not None and n1_label_col is not None:
            n1code_to_label = dict(
                zip(n1[n1_code_col].astype(str), n1[n1_label_col].astype(str))
            )

def apply_job_description_mapping( df: pd.DataFrame, col: str, jd_to_n2code, n2code_to_label, out_col: str):
    s = df[col].astype("string")
    n2code = s.map(lambda x: jd_to_n2code.get(str(x), None) if pd.notna(x) else None)
    
    if n2code_to_label:
        n2label = n2code.map(lambda x: n2code_to_label.get(str(x), str(x)) if pd.notna(x) else None)
        df[out_col] = n2label
    else:
        df[out_col] = n2code
    return df

def main():
    learn = read_csv(LEARN_MASTER)
    test = read_csv(TEST_MASTER)
    
    jd_to_n2code, n2code_to_label = load_mapping_job_description()
    
    if jd_to_n2code is not None:
        if "JOB_DESCRIPTION" in learn.columns:
            learn = apply_job_description_mapping(learn, "JOB_DESCRIPTION", jd_to_n2code, n2code_to_label, "JOB_DESCRIPTION_n2")
            test = apply_job_description_mapping(test, "JOB_DESCRIPTION", jd_to_n2code, n2code_to_label, "JOB_DESCRIPTION_n2")
            
        if "Previous_JOB_DESCRIPTION" in learn.columns:
            learn = apply_job_description_mapping(learn, "Previous_JOB_DESCRIPTION", jd_to_n2code, n2code_to_label, "Previous_JOB_DESCRIPTION_n2")
            test = apply_job_description_mapping(test, "Previous_JOB_DESCRIPTION", jd_to_n2code, n2code_to_label, "Previous_JOB_DESCRIPTION_n2")
        
        drop_raw = [c for c in ["JOB_DESCRIPTION", "Previous_JOB_DESCRIPTION"] if c in learn.columns]
        learn = learn.drop(columns=drop_raw, errors="ignore")
        test = test.drop(columns=drop_raw, errors="ignore")
        
        print ("Applied JOB_DESCRIPTION mapping to n2 and dropped raw columns.")
    else:
        print ("JOB_DESCRIPTION mapping files not found or not understood. Keeping raw codes (may be high cardinality).")

# Save final datasets
    learn.to_csv(ARTIFACTS_DIR / "learn_final.csv", index=False)
    test.to_csv(ARTIFACTS_DIR / "test_final.csv", index=False)
    
    print ("Saved")
    print ("", ARTIFACTS_DIR / "learn_final.csv")
    print ("", ARTIFACTS_DIR / "test_final.csv")
    
if __name__ == "__main__":
    main()
#%%