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