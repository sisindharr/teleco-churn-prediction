"""Central configuration: paths, column groups and business thresholds.

Keeping everything in one file means that the training script, the tests and the
Streamlit app all agree on the same column names and file locations.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA = ROOT / "data" / "processed" / "churn_processed.csv"
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
BASE_MODEL_PATH = ROOT / "models" / "churn_base_pipeline.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
FIGURES_DIR = ROOT / "reports" / "figures"

# ----------------------------------------------------------------------------
# Modelling settings
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
DECISION_THRESHOLD = 0.50  # probability above which we label a customer "will churn"

# ----------------------------------------------------------------------------
# Column groups
# ----------------------------------------------------------------------------
ID_COL = "customerID"
TARGET = "Churn"

# Add-on services that only exist when the customer has internet
ADDON_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Raw text columns (used for whitespace cleaning)
RAW_TEXT_COLS = [
    ID_COL, "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", *ADDON_COLS, "Contract", "PaperlessBilling",
    "PaymentMethod", TARGET,
]

# Features given to the model (after feature engineering)
NUMERIC_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
    "num_addon_services", "monthly_per_service", "has_autopay", "has_family",
]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", *ADDON_COLS, "Contract", "PaperlessBilling",
    "PaymentMethod", "tenure_group",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ----------------------------------------------------------------------------
# Allowed values for the Streamlit input form
# ----------------------------------------------------------------------------
CATEGORY_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes"],
    "InternetService": ["Fiber optic", "DSL", "No"],
    "OnlineSecurity": ["No", "Yes"],
    "OnlineBackup": ["No", "Yes"],
    "DeviceProtection": ["No", "Yes"],
    "TechSupport": ["No", "Yes"],
    "StreamingTV": ["No", "Yes"],
    "StreamingMovies": ["No", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ],
}

# ----------------------------------------------------------------------------
# Risk bands used everywhere in the app (probability of churn)
# ----------------------------------------------------------------------------
LOW_MAX = 0.30    # below this  -> Low risk
HIGH_MIN = 0.60   # at/above    -> High risk (in between -> Medium)
