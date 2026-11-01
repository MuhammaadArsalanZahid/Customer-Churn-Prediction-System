from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from source.config import MODEL_PATH, RAW_FEATURE_COLUMNS, TARGET_COLUMN
from source.data import canonicalize_columns
from source.explain import calculate_shap_values


def load_model_bundle(path=MODEL_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run python scripts/train.py first."
        )

    bundle = joblib.load(path)
    if "pipeline" not in bundle:
        raise ValueError("Saved file does not contain a model pipeline.")
    return bundle


def risk_level(probability):
    if probability < 0.30:
        return "Low"
    if probability < 0.60:
        return "Medium"
    return "High"


def predict_batch(bundle, frame):
    df = canonicalize_columns(frame)
    if df.empty:
        raise ValueError("The uploaded CSV has no customer rows.")

    missing = [c for c in RAW_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing CSV columns: " + ", ".join(missing))

    model_input = df.loc[:, RAW_FEATURE_COLUMNS]
    probabilities = bundle["pipeline"].predict_proba(model_input)[:, 1]
    threshold = float(bundle.get("threshold", 0.5))

    result = df.drop(columns=[TARGET_COLUMN], errors="ignore").copy()
    result["churn_probability"] = probabilities
    result["predicted_churn"] = np.where(
        probabilities >= threshold, "Yes", "No"
    )
    result["risk_level"] = [risk_level(p) for p in probabilities]
    result["decision_threshold"] = threshold
    return result


def customer_shap_reasons(bundle, customer_frame, top_n=3):
    values, names = calculate_shap_values(
        bundle["pipeline"],
        customer_frame,
        background=bundle.get("shap_background"),
    )
    contributions = values[0]
    positive = np.argsort(contributions)[::-1]
    negative = np.argsort(contributions)

    return {
        "risk_factors": [
            {"feature": names[i], "impact": float(contributions[i])}
            for i in positive if contributions[i] > 0
        ][:top_n],
        "protective_factors": [
            {"feature": names[i], "impact": float(contributions[i])}
            for i in negative if contributions[i] < 0
        ][:top_n],
    }


def retention_suggestions(customer):
    suggestions = []

    if str(customer.get("Contract", "")).lower() == "month-to-month":
        suggestions.append(
            "Offer a clear annual-contract option and explain its terms."
        )
    if str(customer.get("TechSupport", "")).lower() == "no":
        suggestions.append(
            "Offer a technical-support check-in or introductory support plan."
        )
    if str(customer.get("PaymentMethod", "")).lower() == "electronic check":
        suggestions.append(
            "Explain available automatic-payment options and their terms."
        )

    try:
        if float(customer.get("tenure", 0)) <= 12:
            suggestions.append(
                "Arrange an early-customer check-in to identify service issues."
            )
    except (TypeError, ValueError):
        pass

    try:
        if float(customer.get("MonthlyCharges", 0)) >= 80:
            suggestions.append(
                "Review the plan and explain suitable lower-cost options, if available."
            )
    except (TypeError, ValueError):
        pass

    if not suggestions:
        suggestions.append(
            "Offer a voluntary check-in before recommending a plan change."
        )
    return suggestions[:4]
