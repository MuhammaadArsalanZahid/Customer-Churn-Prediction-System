import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from source.features import ChurnFeatureEngineer
from source.metrics import choose_business_threshold
from source.modeling import make_pipeline


def toy_data(rows=80):
    records = []
    labels = []

    for i in range(rows):
        records.append({
            "gender": "Female" if i % 2 else "Male",
            "SeniorCitizen": i % 2,
            "Partner": "Yes" if i % 3 else "No",
            "Dependents": "No" if i % 4 else "Yes",
            "tenure": i % 72,
            "PhoneService": "Yes",
            "MultipleLines": "Yes" if i % 2 else "No",
            "InternetService": "Fiber optic" if i % 3 else "DSL",
            "OnlineSecurity": "Yes" if i % 4 else "No",
            "OnlineBackup": "No",
            "DeviceProtection": "Yes",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Month-to-month" if i % 2 else "One year",
            "PaperlessBilling": "Yes",
            "PaymentMethod": (
                "Electronic check" if i % 2 else "Mailed check"
            ),
            "MonthlyCharges": float(40 + i % 70),
            "TotalCharges": np.nan if i == 0 else float(i * 20),
        })
        labels.append(1 if i % 3 == 0 else 0)

    return pd.DataFrame(records), np.asarray(labels)


def test_feature_engineering_creates_features():
    X, _ = toy_data(10)
    result = ChurnFeatureEngineer().fit_transform(X)

    assert "tenure_group" in result
    assert "services_used" in result
    assert "average_charge_per_month" in result
    assert "monthly_charge_per_service" in result


def test_pipeline_returns_valid_probabilities():
    X, y = toy_data()
    model = make_pipeline(
        LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        )
    )
    model.fit(X, y)

    probabilities = model.predict_proba(X.iloc[:5])[:, 1]
    assert probabilities.shape == (5,)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_validation_threshold_is_valid():
    y = np.array([0, 0, 1, 1, 0, 1])
    probabilities = np.array([0.1, 0.3, 0.4, 0.8, 0.2, 0.7])

    threshold, policy = choose_business_threshold(y, probabilities)

    assert 0 <= threshold <= 1
    assert "method" in policy
