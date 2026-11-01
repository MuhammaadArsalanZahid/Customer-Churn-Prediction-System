from pathlib import Path

import numpy as np
import pandas as pd

from source.config import (
    ID_COLUMN,
    NUMERIC_RAW_COLUMNS,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize known dataset column names and surrounding whitespace."""
    result = frame.copy()
    known_names = {
        name.lower(): name
        for name in (*RAW_FEATURE_COLUMNS, TARGET_COLUMN, ID_COLUMN)
    }

    columns = []
    for column in result.columns:
        clean = str(column).replace("\ufeff", "").strip()
        columns.append(known_names.get(clean.lower(), clean))

    if len(columns) != len(set(columns)):
        raise ValueError("CSV contains duplicate column names.")
    result.columns = columns
    return result


def clean_telco_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean Telco data; missing feature values are imputed in the pipeline."""
    df = canonicalize_columns(frame)
    required = [*RAW_FEATURE_COLUMNS, TARGET_COLUMN]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    original_rows = len(df)
    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    for column in df.select_dtypes(include=["object", "string"]).columns:
        values = df[column].astype("string").str.strip()
        values = values.mask(values.eq(""))
        df[column] = values.astype(object).where(values.notna(), np.nan)

    for column in NUMERIC_RAW_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    target = df[TARGET_COLUMN].astype("string").str.strip().str.lower()
    df[TARGET_COLUMN] = target.map({
        "yes": 1, "no": 0, "1": 1, "0": 0,
        "1.0": 1, "0.0": 0, "true": 1, "false": 0,
    })

    invalid_target_rows = int(df[TARGET_COLUMN].isna().sum())
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    summary = {
        "original_rows": int(original_rows),
        "duplicate_rows_removed": duplicate_rows,
        "rows_without_valid_target_removed": invalid_target_rows,
        "clean_rows": int(len(df)),
        "missing_total_charges": int(df["TotalCharges"].isna().sum()),
    }
    return df, summary


def load_telco_data(path: str | Path) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the CSV and put it in data/."
        )
    return clean_telco_data(pd.read_csv(path))
