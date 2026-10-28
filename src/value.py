"""Customer lifetime value and retention-priority calculations."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_customer_value(
    customers: pd.DataFrame,
    gross_margin: float = 0.30,
    expected_remaining_months: int = 24,
) -> pd.DataFrame:
    """Add CLV, expected revenue at risk, and a descending priority rank."""
    if not 0 <= gross_margin <= 1:
        raise ValueError("gross_margin must be between 0 and 1.")
    if expected_remaining_months <= 0:
        raise ValueError("expected_remaining_months must be greater than zero.")
    required = {"MonthlyCharges", "churn_prob"}
    missing = required - set(customers.columns)
    if missing:
        raise ValueError(f"Customer data is missing columns: {sorted(missing)}")
    charges = pd.to_numeric(customers["MonthlyCharges"], errors="raise")
    probabilities = pd.to_numeric(customers["churn_prob"], errors="raise")
    if not np.isfinite(charges).all() or (charges < 0).any():
        raise ValueError("MonthlyCharges must contain finite, non-negative values.")
    if not np.isfinite(probabilities).all() or not probabilities.between(0, 1).all():
        raise ValueError("churn_prob must contain finite values between 0 and 1.")

    valued = customers.copy()
    valued["expected_remaining_months"] = expected_remaining_months
    valued["clv"] = charges * gross_margin * expected_remaining_months
    valued["revenue_at_risk"] = probabilities * valued["clv"]
    valued["priority_rank"] = valued["revenue_at_risk"].rank(
        method="first", ascending=False).astype(int)
    return valued
