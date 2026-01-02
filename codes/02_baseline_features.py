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