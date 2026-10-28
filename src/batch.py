"""In-memory validation and scoring for uploaded customer CSVs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ADDON_COLS, ID_COL
from .data_cleaning import REQUIRED_COLUMNS, clean_data
from .predict import predict_records, risk_band
from .value import add_customer_value

REQUIRED_BATCH_COLUMNS = REQUIRED_COLUMNS - {"Churn"}
RAW_INPUT_COLUMNS = [
    ID_COL, "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", *ADDON_COLS, "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]
NUMERIC_INPUT_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def validate_batch(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate schema and numeric values, then apply training-time data cleaning."""
    data = frame.copy()
    data.columns = data.columns.astype(str).str.strip()
    if data.empty:
        raise ValueError("The CSV contains no customer rows.")
    missing = sorted(REQUIRED_BATCH_COLUMNS - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    for column in NUMERIC_INPUT_COLUMNS:
        original = data[column].astype("string").str.strip()
        numeric = pd.to_numeric(data[column], errors="coerce")
        invalid = numeric.isna() & original.ne("").fillna(False)
        if invalid.any():
            rows = (np.flatnonzero(invalid.to_numpy())[:5] + 2).tolist()
            raise ValueError(f"{column} contains unreadable numbers on CSV row(s): {rows}.")
        if column != "TotalCharges" and numeric.isna().any():
            rows = (np.flatnonzero(numeric.isna().to_numpy())[:5] + 2).tolist()
            raise ValueError(f"{column} is missing a number on CSV row(s): {rows}.")
        if not np.isfinite(numeric.dropna()).all():
            raise ValueError(f"{column} must contain finite numbers.")
        data[column] = numeric
    if not data["SeniorCitizen"].dropna().isin([0, 1]).all():
        raise ValueError("SeniorCitizen values must be 0 or 1.")
    if not data["tenure"].dropna().between(0, 72).all():
        raise ValueError("tenure values must be between 0 and 72 months.")
    if (data["MonthlyCharges"].dropna() < 0).any() or (data["TotalCharges"].dropna() < 0).any():
        raise ValueError("MonthlyCharges and TotalCharges must be non-negative.")

    data = data.drop(columns=["Churn"], errors="ignore")
    return clean_data(data, require_target=False, drop_duplicates=False)


def score_batch(
    model,
    frame: pd.DataFrame,
    gross_margin: float = 0.30,
    expected_remaining_months: int = 24,
) -> pd.DataFrame:
    """Clean and score customer records without persisting the uploaded data."""
    cleaned = validate_batch(frame)
    probabilities = predict_records(model, cleaned.to_dict(orient="records"))
    cleaned["churn_prob"] = probabilities
    cleaned["risk_level"] = cleaned["churn_prob"].map(risk_band)
    scored = add_customer_value(
        cleaned, gross_margin=gross_margin,
        expected_remaining_months=expected_remaining_months)
    scored = scored.rename(columns={"churn_prob": "churn_probability"})
    return scored.sort_values("priority_rank").reset_index(drop=True)


def template_frame() -> pd.DataFrame:
    """A one-row CSV template with realistic raw Telco input columns."""
    row = {
        "customerID": "CUST-0001", "gender": "Female", "SeniorCitizen": 0,
        "Partner": "No", "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "DSL",
        **{column: "No" for column in ADDON_COLS},
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 65.0, "TotalCharges": 780.0,
    }
    return pd.DataFrame([row], columns=RAW_INPUT_COLUMNS)
