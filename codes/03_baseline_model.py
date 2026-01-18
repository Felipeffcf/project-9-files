# Baseline model training script using Random Forest
import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
    precision_recall_curve,
    auc,
)

# Directory paths
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DEFAULT_DATA_DIR = ROOT / "raw_data"
DATA_DIR = Path(os.environ.get("DATA_DIR", DEFAULT_DATA_DIR)).resolve()

ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

FIG_DIR = ARTIFACTS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

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
    learn_uids = learn["uid"].copy()

    y_raw = learn["target"].astype(str)
    X = learn.drop(columns=["target"], errors="ignore")
    X_test = test.copy()

    if "uid" in X.columns:
        X = X.drop(columns=["uid"])
    if "uid" in X_test.columns:
        X_test = X_test.drop(columns=["uid"])

    if SUBSAMPLE_N is not None and SUBSAMPLE_N < len(X):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), size=SUBSAMPLE_N, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y_raw = y_raw.iloc[idx].reset_index(drop=True)
        learn_uids = learn_uids.iloc[idx].reset_index(drop=True)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    preprocessor = build_preprocessor(X)
    rf = RandomForestClassifier(random_state=42, n_jobs=N_JOBS)
    pipe = Pipeline(steps=[("prep", preprocessor), ("rf", rf)])

    # Hyperparameter grid
    param_grid = {
        "rf__n_estimators": [200],
        "rf__max_depth": [None],
        "rf__min_samples_leaf": [1, 5],
        "rf__max_features": ["sqrt"],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Grid search
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

    # Hold-out validation
    idx_all = np.arange(len(X))
    idx_tr, idx_te, ytr, yte = train_test_split(
        idx_all, y, test_size=0.2, stratify=y, random_state=42
    )
    Xtr = X.iloc[idx_tr]
    Xte = X.iloc[idx_te]
    uids_te = learn_uids.iloc[idx_te].reset_index(drop=True)

    best_model.fit(Xtr, ytr)
    pred_hold = best_model.predict(Xte)

    # ROC / AUC on hold-out
    proba_hold = best_model.predict_proba(Xte)[:, 1]
    auc_roc = roc_auc_score(yte, proba_hold)
    print(f"ROC-AUC (hold-out): {auc_roc:.4f}")

    roc_path = FIG_DIR / "roc_curve_holdout.png"
    RocCurveDisplay.from_predictions(yte, proba_hold)
    plt.title("ROC Curve (Hold-out)")
    plt.savefig(roc_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curve: {roc_path}")

    prec, rec, _ = precision_recall_curve(yte, proba_hold)
    pr_auc = auc(rec, prec)
    print(f"PR-AUC (hold-out): {pr_auc:.4f}")

    pr_path = FIG_DIR / "pr_curve_holdout.png"
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (Hold-out)")
    plt.savefig(pr_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved PR curve: {pr_path}")

    # Evaluation metrics
    acc = accuracy_score(yte, pred_hold)
    f1m = f1_score(yte, pred_hold, average="macro")
    cmat = confusion_matrix(yte, pred_hold)
    rep = classification_report(yte, pred_hold)

    print("\n=== Hold-out validation results ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1m:.4f}")
    print("\nConfusion matrix:")
    print(cmat)
    print("\nClassification report:")
    print(rep)

    # Confident wrong predictions exploration
    pred_from_proba = (proba_hold >= 0.5).astype(int)
    wrong = pred_from_proba != yte
    wrong_idx = np.where(wrong)[0]

    confidence = np.where(pred_from_proba == 1, proba_hold, 1.0 - proba_hold)
    wrong_conf = confidence[wrong_idx]

    topk = min(30, len(wrong_idx))
    if topk > 0:
        rank = np.argsort(wrong_conf)[::-1][:topk]
        hard_idx = wrong_idx[rank]

        hard_df = pd.DataFrame(
            {
                "uid": uids_te.iloc[hard_idx].values,
                "y_true": yte[hard_idx],
                "y_pred": pred_from_proba[hard_idx],
                "proba_class1": proba_hold[hard_idx],
                "confidence": confidence[hard_idx],
            }
        )

        hard_path = ARTIFACTS_DIR / "most_confident_mistakes_holdout.csv"
        hard_df.to_csv(hard_path, index=False)
        print(f"Saved confident mistakes table: {hard_path}")
    else:
        print("No misclassifications found on the hold-out set (unexpected, but possible).")

    # Final training on full data and prediction
    best_model.fit(X, y)
    pred_test_enc = best_model.predict(X_test)
    pred_test_labels = le.inverse_transform(pred_test_enc)

    pred_df = pd.DataFrame({"uid": test_uids, "target": pred_test_labels})
    pred_df.to_csv(PRED_PATH, index=False)
    print(">>> SAVING PREDICTIONS NOW <<<")
    print(f"Saved to: {PRED_PATH.resolve()}")
    print(f"Rows: {len(pred_df)}")

    joblib.dump(best_model, ARTIFACTS_DIR / "model.joblib")
    joblib.dump(le, ARTIFACTS_DIR / "label_encoder.joblib")

    # Feature importance (Random Forest)
    prep = best_model.named_steps["prep"]
    rf_model = best_model.named_steps["rf"]

    num_cols = prep.transformers_[0][2]
    cat_cols = prep.transformers_[1][2]
    ohe = prep.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = ohe.get_feature_names_out(cat_cols)

    feature_names = np.array(list(num_cols) + list(cat_feature_names))
    importances = rf_model.feature_importances_

    idx = np.argsort(importances)[::-1][:20]
    imp_df = pd.DataFrame({"feature": feature_names[idx], "importance": importances[idx]})

    imp_path = ARTIFACTS_DIR / "feature_importance_top20.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"Saved feature importance table: {imp_path}")

    imp_plot_path = FIG_DIR / "feature_importance_top20.png"
    plt.figure()
    plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
    plt.title("Top-20 Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.savefig(imp_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved feature importance plot: {imp_plot_path}")


# End of main()
if __name__ == "__main__":
    main()
