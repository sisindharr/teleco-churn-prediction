"""Prediction helpers shared by the app and the tests.

Contains: risk banding, single/batch prediction, "what-if" scenarios and
rule-based retention recommendations.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .config import ADDON_COLS, ALL_FEATURES, HIGH_MIN, LOW_MAX
from .features import add_features


# ----------------------------------------------------------------------------
# Risk bands
# ----------------------------------------------------------------------------
def risk_band(prob: float) -> str:
    """Convert a churn probability into Low / Medium / High."""
    if prob >= HIGH_MIN:
        return "High"
    if prob >= LOW_MAX:
        return "Medium"
    return "Low"


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
def normalize_records(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same business rules used in cleaning to user-entered data."""
    df = df.copy()
    no_internet = df["InternetService"] == "No"
    for col in ADDON_COLS:
        df.loc[no_internet, col] = "No"
    df.loc[df["PhoneService"] == "No", "MultipleLines"] = "No"
    if "TotalCharges" not in df.columns:
        df["TotalCharges"] = df["MonthlyCharges"] * df["tenure"]
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    return df


def predict_records(model, records: Iterable[dict]) -> np.ndarray:
    """Return churn probabilities for one or more customer dictionaries."""
    df = normalize_records(pd.DataFrame(list(records)))
    df = add_features(df)
    return model.predict_proba(df[ALL_FEATURES])[:, 1]


# ----------------------------------------------------------------------------
# What-if scenarios (associations learned by the model, not guaranteed causes)
# ----------------------------------------------------------------------------
SCENARIOS: dict[str, dict[str, str]] = {
    "Move to a 1-year contract": {"Contract": "One year"},
    "Move to a 2-year contract": {"Contract": "Two year"},
    "Switch to automatic card payment": {"PaymentMethod": "Credit card (automatic)"},
    "Add tech support": {"TechSupport": "Yes"},
    "Add online security": {"OnlineSecurity": "Yes"},
}


def what_if(model, record: dict, base_prob: float) -> list[dict]:
    """Estimate how churn risk changes under simple retention offers."""
    results = []
    for name, change in SCENARIOS.items():
        # Skip scenarios that do not apply or would change nothing
        if all(record.get(k) == v for k, v in change.items()):
            continue
        if ("TechSupport" in change or "OnlineSecurity" in change) and record["InternetService"] == "No":
            continue
        new_prob = float(predict_records(model, [{**record, **change}])[0])
        results.append({"scenario": name, "new_prob": new_prob, "delta": new_prob - base_prob})
    return sorted(results, key=lambda r: r["delta"])


# ----------------------------------------------------------------------------
# Rule-based recommendations
# ----------------------------------------------------------------------------
def recommendations(record: dict, prob: float) -> list[dict]:
    """Human-readable next-best-actions for the retention team."""
    recs: list[dict] = []

    def add(priority: str, title: str, detail: str) -> None:
        recs.append({"priority": priority, "title": title, "detail": detail})

    if prob >= HIGH_MIN:
        add("High", "Escalate to a retention specialist",
            "Call the customer within 48 hours and offer a personalised deal before they cancel.")
    if record["Contract"] == "Month-to-month":
        add("High", "Offer a longer contract",
            "Month-to-month customers can leave at any time. A discount or free month for a 12-month plan raises commitment.")
    if record["InternetService"] == "Fiber optic" and record["TechSupport"] == "No":
        add("Medium", "Bundle tech support",
            "Fiber customers without support are more likely to leave after a service issue. Offer a free 3-month trial.")
    if record["InternetService"] != "No" and record["OnlineSecurity"] == "No":
        add("Medium", "Promote online security",
            "Security add-ons increase stickiness. Offer a trial bundled with the next bill.")
    if record["PaymentMethod"] == "Electronic check":
        add("Medium", "Encourage automatic payments",
            "Give a small bill credit for switching to card or bank auto-pay. It removes a monthly decision point.")
    if record["tenure"] <= 12:
        add("Medium", "Strengthen onboarding",
            "Customers in their first year churn most. Schedule check-ins at 30, 60 and 90 days.")
    if record["MonthlyCharges"] >= 80:
        add("Low", "Review the plan price",
            "High bills drive cancellations. Check whether a cheaper plan or a loyalty discount fits their usage.")
    if not recs:
        add("Low", "Keep nurturing the relationship",
            "This customer looks stable. Share loyalty perks and ask for a referral or review.")
    order = {"High": 0, "Medium": 1, "Low": 2}
    return sorted(recs, key=lambda r: order[r["priority"]])
