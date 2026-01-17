#%% 
import os
from pathlib import Path

from matplotlib import cm
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DEFAULT_DATA_DIR = ROOT / "raw_data"
DATA_DIR = Path(os.environ.get("DATA_DIR", DEFAULT_DATA_DIR)).resolve()

ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

LEARN_MASTER_PATH = ARTIFACTS_DIR / "learn_master.csv"
TEST_MASTER_PATH = ARTIFACTS_DIR / "test_master.csv"

PRED_PATH = ROOT / "predictions.csv"

SUBSAMPLE_N = None
N_JOBS = -1
VERBOSE = 1


def must_exist(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def load_masters() -> tuple[pd.DataFrame, pd.DataFrame]:
    dtype_map = {
        "INSEE": str,
        "DEP": str,
        "JOB_DEP_x": str,
        "JOB_DEP_y": str,
        "PREVIOUS_DEP": str,
    }
    learn = pd.read_csv(must_exist(LEARN_MASTER_PATH), low_memory=False, dtype=dtype_map)
    test = pd.read_csv(must_exist(TEST_MASTER_PATH), low_memory=False, dtype=dtype_map)
    return learn, test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def main():
    learn, test = load_masters()

    if "uid" not in learn.columns or "uid" not in test.columns:
        raise ValueError("'uid' column missing.")
    if "target" not in learn.columns:
        raise ValueError("'target' column missing in learn data.")

    test_uids = test["uid"].copy()

    y_raw = learn["target"].astype(str)
    X = learn.drop(columns=["target"], errors="ignore")
    if "uid" in X.columns:
        X = X.drop(columns=["uid"])

    X_test = test.copy()
    if "uid" in X_test.columns:
        X_test = X_test.drop(columns=["uid"])

    if SUBSAMPLE_N is not None and SUBSAMPLE_N < len(X):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), size=SUBSAMPLE_N, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y_raw = y_raw.iloc[idx].reset_index(drop=True)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    preprocessor = build_preprocessor(X)
    rf = RandomForestClassifier(random_state=42, n_jobs=N_JOBS)
    pipe = Pipeline(steps=[("prep", preprocessor), ("rf", rf)])

    param_grid = {
        "rf__n_estimators": [200],
        "rf__max_depth": [None],
        "rf__min_samples_leaf": [1, 5],
        "rf__max_features": ["sqrt"],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=N_JOBS,
        verbose=VERBOSE,
        error_score="raise",
    )

    gs.fit(X, y)
    best_model = gs.best_estimator_
    print(f"Best CV macro-F1: {gs.best_score_:.4f}")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    best_model.fit(Xtr, ytr)
    pred_hold = best_model.predict(Xte)

    acc = accuracy_score(yte, pred_hold)
    f1m = f1_score(yte, pred_hold, average="macro")
    cm = confusion_matrix(yte, pred_hold)
    rep = classification_report(yte, pred_hold)

    print("\n=== Hold-out validation results ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1m:.4f}")
    print("\nConfusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(rep)

    best_model.fit(X, y)
    pred_test_enc = best_model.predict(X_test)
    pred_test_labels = le.inverse_transform(pred_test_enc)

    pred_df = pd.DataFrame({"uid": test_uids, "target": pred_test_labels})
    pred_df.to_csv(PRED_PATH, index=False)

    joblib.dump(best_model, ARTIFACTS_DIR / "model.joblib")
    joblib.dump(le, ARTIFACTS_DIR / "label_encoder.joblib")


if __name__ == "__main__":
    main()
# %%