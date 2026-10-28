"""Step 1 - Data cleaning.

Real-world data is messy. This module turns the raw Kaggle CSV into a tidy table
that machine-learning models (and humans) can use.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import ADDON_COLS, RAW_DATA, RAW_TEXT_COLS, TARGET

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = set(RAW_TEXT_COLS) | {"SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"}


def load_raw(path: Path | str = RAW_DATA) -> pd.DataFrame:
    """Read the raw CSV and fail early (with a helpful message) if it is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download 'Telco Customer Churn' from Kaggle "
            "(https://www.kaggle.com/datasets/blastchar/telco-customer-churn) "
            "and place the CSV in data/raw/."
        )
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame, require_target: bool = True) -> None:
    """Make sure all expected columns are present."""
    required = REQUIRED_COLUMNS if require_target else REQUIRED_COLUMNS - {TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")


def clean_data(
    df: pd.DataFrame,
    require_target: bool = True,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """Return a cleaned copy of the raw Telco data.

    What we fix and why
    -------------------
    1. Remove duplicate rows              -> duplicates would bias the model.
    2. TotalCharges is text, not a number -> 11 rows contain a blank " " (new customers
       with tenure 0). We convert to numbers and set those blanks to 0, because a
       customer who has not been billed yet has paid 0 so far.
    3. "No internet service" / "No phone service" -> these mean the same as "No" for
       the related add-on. Merging them removes redundant categories.
    4. Target "Yes"/"No" -> 1/0 so the model can learn from it.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    validate_schema(df, require_target=require_target)

    n_before = len(df)
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < n_before:
        logger.info("Dropped %d duplicate rows", n_before - len(df))

    # Strip stray whitespace in text columns
    for col in RAW_TEXT_COLS:
        if col not in df:
            continue
        df[col] = df[col].astype(str).str.strip()

    # TotalCharges: text -> number
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    new_customer = df["TotalCharges"].isna() & (df["tenure"] == 0)
    df.loc[new_customer, "TotalCharges"] = 0.0
    # Any remaining gap: estimate as tenure x monthly price
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])

    # Merge "No internet service" / "No phone service" into "No"
    for col in [*ADDON_COLS, "MultipleLines"]:
        df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})

    # Target to 0/1
    if TARGET in df:
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
        if df[TARGET].isna().any():
            raise ValueError("Churn column contains values other than 'Yes'/'No'.")
        df[TARGET] = df[TARGET].astype(int)
    elif require_target:
        raise ValueError(f"Dataset is missing expected columns: ['{TARGET}']")

    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    return df
