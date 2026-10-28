"""Step 2 - Feature engineering.

Feature engineering = creating new columns that make patterns easier for a model
to see. The SAME function is used in training and in the Streamlit app so that
predictions are consistent (no "training-serving skew").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ADDON_COLS, ALL_FEATURES

TENURE_BINS = [-1, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-12 months", "13-24 months", "25-48 months", "49+ months"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add business-driven features to a cleaned dataframe."""
    df = df.copy()

    # How long has the customer been with us? (new customers churn the most)
    df["tenure_group"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)

    # How "sticky" is the customer? More add-on services = harder to leave
    df["num_addon_services"] = (df[ADDON_COLS] == "Yes").sum(axis=1).astype(int)

    # Are they paying a lot for what they get?
    df["monthly_per_service"] = df["MonthlyCharges"] / (df["num_addon_services"] + 1)

    # Automatic payments reduce friction -> fewer cancellations
    df["has_autopay"] = df["PaymentMethod"].str.contains("automatic", case=False).astype(int)

    # Customers with a partner/dependents tend to be more stable
    df["has_family"] = ((df["Partner"] == "Yes") | (df["Dependents"] == "Yes")).astype(int)
    return df


def split_xy(df: pd.DataFrame, target: str = "Churn") -> tuple[pd.DataFrame, pd.Series]:
    """Return model inputs (X) and the target column (y)."""
    return df[ALL_FEATURES], df[target]
