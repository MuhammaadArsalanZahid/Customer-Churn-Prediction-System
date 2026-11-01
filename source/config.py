from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT_DIR / "models"

DATA_PATH = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

RAW_FEATURE_COLUMNS = (
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
)

NUMERIC_RAW_COLUMNS = (
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
)

CATEGORICAL_RAW_COLUMNS = tuple(
    column for column in RAW_FEATURE_COLUMNS
    if column not in NUMERIC_RAW_COLUMNS
)

ENGINEERED_FEATURE_COLUMNS = (
    "tenure_group",
    "services_used",
    "average_charge_per_month",
    "monthly_charge_per_service",
)
