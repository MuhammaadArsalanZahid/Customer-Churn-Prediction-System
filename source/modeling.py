import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from source.features import ChurnFeatureEngineer


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_one_hot_encoder()),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, make_column_selector(dtype_include=np.number)),
            (
                "categorical",
                categorical,
                make_column_selector(dtype_exclude=np.number),
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def make_pipeline(classifier):
    return Pipeline([
        ("features", ChurnFeatureEngineer()),
        ("preprocessor", make_preprocessor()),
        ("model", classifier),
    ])


def model_catalog(y_train):
    from xgboost import XGBClassifier

    y = np.asarray(y_train).astype(int)
    positives = max(int((y == 1).sum()), 1)
    negatives = int((y == 0).sum())
    positive_weight = negatives / positives

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=positive_weight,
            tree_method="hist",
            n_jobs=1,
            random_state=42,
        ),
    }


def parameter_grids():
    return {
        "Logistic Regression": {
            "model__C": [0.1, 1.0, 5.0],
        },
        "Decision Tree": {
            "model__max_depth": [None, 6, 10],
            "model__min_samples_leaf": [1, 5],
        },
        "Random Forest": {
            "model__n_estimators": [150, 300],
            "model__max_depth": [None, 12],
            "model__min_samples_leaf": [1, 3],
        },
        "Gradient Boosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        },
        "XGBoost": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [2, 4],
            "model__learning_rate": [0.03, 0.1],
        },
    }
