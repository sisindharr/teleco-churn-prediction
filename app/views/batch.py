"""Batch CSV scoring page."""
from __future__ import annotations

import logging

import pandas as pd
import plotly.express as px
import streamlit as st

import components as ui
import data_access as da
from compat import plotly, stretch
from src.batch import score_batch, template_frame
from src.config import HIGH_MIN
from theme import RISK_COLORS, style_fig

logger = logging.getLogger(__name__)


def render(ctx: da.Ctx) -> None:
    ui.page_header("Batch scoring", "Upload a customer CSV to score and prioritize retention outreach.")
    ui.md('<div class="card-text">Upload data is validated and scored in memory only; it is not saved to disk or a database.</div>')
    template = template_frame().to_csv(index=False).encode("utf-8")
    stretch(st.download_button, "Download CSV template", template,
            file_name="telco_customer_template.csv", mime="text/csv", key="batch_template")
    uploaded = st.file_uploader("Upload customer CSV (up to 20 MB)", type=["csv"],
                                max_upload_size=20, key="batch_upload")
    if uploaded is None:
        return

    gross_margin = st.slider("Gross margin", 5, 100, 30, 5, format="%d%%", key="batch_margin") / 100
    expected_months = st.slider("Expected remaining months", 1, 60, 24, key="batch_months")
    try:
        with st.spinner("Validating and scoring customers..."):
            try:
                frame = pd.read_csv(uploaded)
            except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                raise ValueError(f"Could not read this file as a CSV: {exc}") from exc
            scored = score_batch(ctx.model, frame, gross_margin, expected_months)
    except ValueError as exc:
        st.error(str(exc))
        return
    except Exception:
        logger.exception("Batch scoring failed")
        st.error("Batch scoring failed unexpectedly. Please check the CSV format and try again.")
        return

    high_count = int((scored["risk_level"] == "High").sum())
    k = st.columns(3)
    with k[0]:
        ui.kpi_card("Rows scored", f"{len(scored):,}", "customer records", "👥")
    with k[1]:
        ui.kpi_card("High-risk customers", f"{high_count:,}",
                    f"risk probability at or above {HIGH_MIN:.0%}", "🚨", "#EF4444")
    with k[2]:
        ui.kpi_card("Total revenue at risk", ui.money(scored["revenue_at_risk"].sum()),
                    f"{gross_margin:.0%} margin, {expected_months} months", "💰", "#F59E0B")

    counts = scored["risk_level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    distribution = counts.rename_axis("Risk level").reset_index(name="Customers")
    fig = px.bar(distribution, x="Risk level", y="Customers", color="Risk level",
                 color_discrete_map=RISK_COLORS)
    plotly(style_fig(fig, ctx.dark, legend=False, title="Risk-level distribution"), key="batch_risk_distribution")

    ui.section("Top 20 by revenue at risk")
    columns = ["customerID", "churn_probability", "risk_level", "MonthlyCharges",
               "revenue_at_risk", "priority_rank"]
    table = scored[columns].head(20).rename(columns={
        "customerID": "Customer", "churn_probability": "Churn probability",
        "risk_level": "Risk", "MonthlyCharges": "Monthly bill",
        "revenue_at_risk": "Revenue at risk", "priority_rank": "Priority rank",
    }).copy()
    table["Churn probability"] = table["Churn probability"].map(lambda value: f"{value:.0%}")
    table["Monthly bill"] = table["Monthly bill"].map(ui.money)
    table["Revenue at risk"] = table["Revenue at risk"].map(ui.money)
    ui.html_table(table, risk_col="Risk")
    stretch(st.download_button, "Download scored CSV",
            scored.to_csv(index=False).encode("utf-8"), file_name="telco_churn_scored.csv",
            mime="text/csv", key="batch_download")
