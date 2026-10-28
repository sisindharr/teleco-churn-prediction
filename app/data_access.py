"""Cached loading of the model, metrics and data (so pages stay fast)."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import ALL_FEATURES, METRICS_PATH, MODEL_PATH, PROCESSED_DATA  # noqa: E402
from src.predict import risk_band  # noqa: E402


@dataclass
class Ctx:
    """Everything a page needs, passed from main.py."""
    model: object
    metrics: dict
    data: pd.DataFrame
    dark: bool


def artifacts_ready() -> bool:
    return all(p.exists() for p in (MODEL_PATH, METRICS_PATH, PROCESSED_DATA))


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    return json.loads(Path(METRICS_PATH).read_text())


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DATA)
    df["ChurnLabel"] = df["Churn"].map({1: "Churned", 0: "Retained"})
    return df


@st.cache_data(show_spinner="Scoring customers...")
def score_all(_model, df: pd.DataFrame) -> pd.Series:
    """Churn probability for every customer in the dataset."""
    proba = _model.predict_proba(df[ALL_FEATURES])[:, 1]
    return pd.Series(proba, index=df.index, name="churn_prob")


def scored_data(ctx: Ctx) -> pd.DataFrame:
    probs = score_all(ctx.model, ctx.data)
    return ctx.data.assign(churn_prob=probs, risk=probs.map(risk_band))


def churn_rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Churn rate (%) and customer count for each value of `col`."""
    out = (df.groupby(col, observed=True)["Churn"].agg(rate="mean", customers="size").reset_index())
    out["rate"] = out["rate"] * 100
    return out
