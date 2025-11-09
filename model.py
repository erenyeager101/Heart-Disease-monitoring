# heart_pipeline.py
"""
Optimized heart disease classification pipeline:
- loads data
- builds preprocessing pipelines for numeric + categorical features
- uses RandomizedSearchCV to tune RandomForest + HistGradientBoosting
- builds a StackingClassifier
- evaluates using StratifiedKFold cross-validation and holdout test
- saves final pipeline with joblib
"""

import os
import warnings
from typing import Tuple, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, train_test_split,
                                     cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---- Config / reproducibility ----
RANDOM_STATE = 42
N_JOBS = -1
CV = 5
RANDOM_SEARCH_ITERS = 40  # smaller for speed, increase if you have time

warnings.filterwarnings("ignore")


def load_data(path: str = "heart.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path)
    return df


def inspect_dataset(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nMissing values per column:\n", df.isna().sum())
    print("\nValue counts for possible categorical columns (first few):")
    for col in df.columns:
        if df[col].nunique() <= 10:
            print(f"  {col}: {df[col].unique()[:10]} (unique={df[col].nunique()})")


def infer_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Try to infer numeric vs categorical features.
    We'll treat low-cardinality integer columns as categorical (<=10 unique values)
    and floats/high-cardinality ints as numeric.
    """
    excluded = {"target"}
    numeric_cols = []
    categorical_cols = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_float_dtype(df[col]):
            numeric_cols.append(col)
        elif pd.api.types.is_integer_dtype(df[col]):
            if df[col].nunique() <= 10:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    return preprocessor


def random_search_for_model(pipeline: Pipeline, param_distributions: dict, X, y, iters=20):
    rs = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=iters,
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=CV, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=1
    )
    rs.fit(X, y)
    print(f"Best score: {rs.best_score_:.4f}; Best params: {rs.best_params_}")
    return rs.best_estimator_


def evaluate_final(pipeline: Pipeline, X_test, y_test) -> None:
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()


def main():
    df = load_data("heart.csv")
    inspect_dataset(df)

    if "target" not in df.columns:
        raise KeyError("The dataset must contain a 'target' column.")

    numeric_features, categorical_features = infer_feature_types(df)
    print("\nNumeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # ---- Candidate models pipelines for RandomizedSearch ----
    # Random Forest pipeline
    rf_pipeline = Pipeline([
        ("preproc", preprocessor),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1))   # n_jobs set globally in RandomizedSearchCV
    ])

    rf_param_dist = {
        "clf__n_estimators": [100, 200, 400, 800],
        "clf__max_depth": [None, 6, 8, 12],
        "clf__min_samples_split": [2, 4, 8],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__bootstrap": [True, False],
        "clf__class_weight": [None, "balanced"]
    }

    # HistGradientBoosting pipeline (fast and often very good)
    hgb_pipeline = Pipeline([
        ("preproc", preprocessor),
        ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE))
    ])

    hgb_param_dist = {
        "clf__max_iter": [100, 200, 400],
        "clf__max_leaf_nodes": [15, 31, 63, None],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "clf__l2_regularization": [0.0, 0.1, 0.5]
    }

    # Run randomized search (speed / performance balance)
    print("\nTuning RandomForest (RandomizedSearchCV)...")
    best_rf = random_search_for_model(rf_pipeline, rf_param_dist, X_train, y_train, iters=min(RANDOM_SEARCH_ITERS, 40))

    print("\nTuning HistGradientBoosting (RandomizedSearchCV)...")
    best_hgb = random_search_for_model(hgb_pipeline, hgb_param_dist, X_train, y_train, iters=min(RANDOM_SEARCH_ITERS, 24))

    # ---- Build stacking classifier using the tuned estimators ----
    estimators = [
        ("rf", best_rf.named_steps["clf"]),
        ("hgb", best_hgb.named_steps["clf"])
    ]

    # We need a pipeline that contains preprocessing and the stacking classifier
    stacking_pipeline = Pipeline([
        ("preproc", preprocessor),
        ("stack", StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=2000),
            n_jobs=N_JOBS,
            passthrough=False
        ))
    ])

    print("\nEvaluating stacking pipeline with cross-validation...")
    skf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(stacking_pipeline, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=N_JOBS)
    print(f"Cross-validated accuracy (stacking): mean={cv_scores.mean():.4f}, std={cv_scores.std():.4f}")

    # Fit on full training set and evaluate on test set
    stacking_pipeline.fit(X_train, y_train)
    evaluate_final(stacking_pipeline, X_test, y_test)

    # Save the final pipeline
    model_path = "heart_stacking_pipeline.joblib"
    joblib.dump(stacking_pipeline, model_path)
    print(f"\nSaved final pipeline to {model_path}")

    # Example prediction (replace values with your feature order)
    sample = X_test.iloc[0:1]
    print("\nSample input (first test row):")
    print(sample)
    print("Predicted class:", stacking_pipeline.predict(sample))
    if hasattr(stacking_pipeline, "predict_proba"):
        try:
            print("Predicted probabilities:", stacking_pipeline.predict_proba(sample))
        except Exception:
            pass


if __name__ == "__main__":
    main()
