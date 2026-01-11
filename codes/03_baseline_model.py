#%% 
import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

LEARN_MASTER = Path(os.environ.get("LEARN_MASTER", ARTIFACTS_DIR / "learn_master.csv")).resolve()
TEST_MASTER  = Path(os.environ.get("TEST_MASTER",  ARTIFACTS_DIR / "test_master.csv")).resolve()
PRED_PATH    = Path(os.environ.get("PRED_PATH", ROOT / "predictions.csv")).resolve()

# Debug knobs
SUBSAMPLE_N = 15000   # set None to use full data
N_JOBS = -1           # -1 uses all cores
VERBOSE = 2           # GridSearch verbosity

# If your positive class in the original data is 'X', keep this to output 0/1 predictions.
# If None, will output the original labels via inverse_transform.
POSITIVE_LABEL = "X"


def make_ohe():
    # Compatibility across scikit-learn versions
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline([
        ("to_str", FunctionTransformer(lambda x: x.astype(str))),
        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("onehot", make_ohe()),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


def main():
    if not LEARN_MASTER.exists():
        raise FileNotFoundError(f"Learn master dataset not found: {LEARN_MASTER}")
    if not TEST_MASTER.exists():
        raise FileNotFoundError(f"Test master dataset not found: {TEST_MASTER}")

    learn = pd.read_csv(LEARN_MASTER, dtype={"INSEE": str})
    test  = pd.read_csv(TEST_MASTER,  dtype={"INSEE": str})

    if "target" not in learn.columns:
        raise ValueError("learn_master.csv must contain 'target' column.")
    if "uid" not in test.columns:
        raise ValueError("test_master.csv must contain 'uid' column.")

    # Encode target labels -> 0..K-1 (handles cases like 'X')
    y_raw = learn["target"].astype(str)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print("Target classes:", list(le.classes_), flush=True)
    if POSITIVE_LABEL is not None and POSITIVE_LABEL not in list(le.classes_):
        print(
            f"WARNING: POSITIVE_LABEL='{POSITIVE_LABEL}' not found in classes. "
            "Predictions will be saved as original labels (not 0/1).",
            flush=True
        )

    # Features
    X = learn.drop(columns=["target", "uid"], errors="ignore")
    X_test = test.drop(columns=["uid"], errors="ignore")
    test_uids = test["uid"].copy()

    # Subsample for faster debugging (optional)
    if SUBSAMPLE_N is not None and SUBSAMPLE_N < len(X):
        idx = X.sample(n=SUBSAMPLE_N, random_state=42).index
        X = X.loc[idx]
        y = y[idx]
        print(f"Using subsample n={SUBSAMPLE_N}. X shape: {X.shape}", flush=True)

    pre = make_preprocessor(X)

    clf = RandomForestClassifier(
        random_state=42,
        n_jobs=N_JOBS,
        class_weight="balanced_subsample",
    )

    pipe = Pipeline(steps=[
        ("pre", pre),
        ("rf", clf),
    ])

    # ✅ Reduced grid to avoid extremely long runtimes
    param_grid = {
        "rf__n_estimators": [300],
        "rf__max_depth": [None, 20],
        "rf__min_samples_leaf": [1, 5],
        "rf__max_features": ["sqrt", 0.7],
    }

    # ✅ Reduced folds to speed up while still using resampling (mandatory)
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

    print("Starting GridSearchCV...", flush=True)
    gs.fit(X, y)

    print("\nBest CV f1_macro:", gs.best_score_, flush=True)
    print("Best params:", gs.best_params_, flush=True)
    best_model = gs.best_estimator_

    # Holdout check (optional, for sanity)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    best_model.fit(Xtr, ytr)
    pred = best_model.predict(Xte)

    print("\nHoldout accuracy:", accuracy_score(yte, pred), flush=True)
    print("Holdout f1_macro:", f1_score(yte, pred, average="macro"), flush=True)

    cm = confusion_matrix(yte, pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print("\nConfusion matrix:\n", cm_df, flush=True)

    print("\nClassification report:\n", classification_report(yte, pred, target_names=le.classes_), flush=True)

    # Fit on all learn data for final predictions
    best_model.fit(X, y)
    pred_test_enc = best_model.predict(X_test)
    pred_test_labels = le.inverse_transform(pred_test_enc)

    # Output format: 0/1 if POSITIVE_LABEL is set and exists in classes, else output original labels
    if POSITIVE_LABEL is not None and POSITIVE_LABEL in list(le.classes_):
        pred_test_out = (pred_test_labels == POSITIVE_LABEL).astype(int)
    else:
        pred_test_out = pred_test_labels

    pred_df = pd.DataFrame({"uid": test_uids, "target": pred_test_out})
    pred_df.to_csv(PRED_PATH, index=False)
    print(f"\nSaved predictions to: {PRED_PATH}", flush=True)

    # Save artifacts
    joblib.dump(best_model, ARTIFACTS_DIR / "model.joblib")
    joblib.dump(le, ARTIFACTS_DIR / "label_encoder.joblib")
    print(f"Saved model and label encoder to: {ARTIFACTS_DIR}", flush=True)


if __name__ == "__main__":
    main()

# %%
