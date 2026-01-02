#%%

import os
import time
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

DATA_LEARN = r"C:\Users\felif\Desktop\Machine Learning\FINAL PROJECT\project-9-files\raw_data\learn_dataset.csv"

N_JOBS = 4
VERBOSE = 2

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    return ColumnTransformer(
        [("num", numeric_pipe, numeric_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
    )

def main():
    print("=== START full baseline ===", flush=True)
    print("CWD:", os.getcwd(), flush=True)
    print("Reading:", DATA_LEARN, flush=True)

    t0 = time.time()
    learn = pd.read_csv(DATA_LEARN, dtype={"INSEE": str})
    print("Loaded learn:", learn.shape, "in", round(time.time() - t0, 2), "sec", flush=True)

    X = learn.drop(columns=["target"])
    y_raw = learn["target"].astype(str)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print("Classes:", list(le.classes_), flush=True)
    print("Counts :", dict(zip(*np.unique(y, return_counts=True))), flush=True)

    pre = make_preprocessor(X)

    pipe = Pipeline([
        ("pre", pre),
        ("rf", RandomForestClassifier(
            random_state=42,
            n_jobs=N_JOBS,
            class_weight="balanced_subsample",
        )),
    ])

    # Grid pequeño para full data (más llevadero)
    param_grid = {
        "rf__n_estimators": [300],
        "rf__max_depth": [20, None],
        "rf__min_samples_leaf": [1, 5],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=N_JOBS,
        verbose=VERBOSE,
        error_score="raise",
    )

    print("Fitting GridSearchCV... (this can take a while)", flush=True)
    t1 = time.time()
    gs.fit(X, y)
    print("Done fit in", round(time.time() - t1, 2), "sec", flush=True)

    print("\nBest CV f1_macro:", gs.best_score_, flush=True)
    print("Best params:", gs.best_params_, flush=True)
    print("=== END ===", flush=True)

if __name__ == "__main__":
    main()

#%%