import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from source.config import (
    CATEGORICAL_RAW_COLUMNS,
    ENGINEERED_FEATURE_COLUMNS,
    NUMERIC_RAW_COLUMNS,
    RAW_FEATURE_COLUMNS,
)


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create customer features as part of the sklearn pipeline."""

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _yes(series):
        values = series.astype("string").str.strip().str.lower()
        return values.eq("yes").fillna(False).astype(np.int8)

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=RAW_FEATURE_COLUMNS)

        for column in RAW_FEATURE_COLUMNS:
            if column not in df.columns:
                df[column] = np.nan

        for column in NUMERIC_RAW_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        for column in CATEGORICAL_RAW_COLUMNS:
            values = df[column].astype("string").str.strip()
            values = values.mask(values.eq(""))
            df[column] = values.astype(object).where(values.notna(), np.nan)

        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[-np.inf, 12, 24, 48, 60, np.inf],
            labels=[
                "0-12 months", "13-24 months", "25-48 months",
                "49-60 months", "61+ months",
            ],
        ).astype(object)

        internet = (
            df["InternetService"]
            .astype("string")
            .str.strip()
            .str.lower()
            .isin(["dsl", "fiber optic"])
            .astype(np.int8)
        )

        services = pd.DataFrame({
            "PhoneService": self._yes(df["PhoneService"]),
            "MultipleLines": self._yes(df["MultipleLines"]),
            "InternetService": internet,
            "OnlineSecurity": self._yes(df["OnlineSecurity"]),
            "OnlineBackup": self._yes(df["OnlineBackup"]),
            "DeviceProtection": self._yes(df["DeviceProtection"]),
            "TechSupport": self._yes(df["TechSupport"]),
            "StreamingTV": self._yes(df["StreamingTV"]),
            "StreamingMovies": self._yes(df["StreamingMovies"]),
        }, index=df.index)

        df["services_used"] = services.sum(axis=1).astype(float)

        average = df["TotalCharges"].div(
            df["tenure"].where(df["tenure"] > 0)
        )
        df["average_charge_per_month"] = average.where(
            average.notna(), df["MonthlyCharges"]
        )
        df["monthly_charge_per_service"] = df["MonthlyCharges"].div(
            df["services_used"].clip(lower=1)
        )

        return df[
            [*RAW_FEATURE_COLUMNS, *ENGINEERED_FEATURE_COLUMNS]
        ].copy()
