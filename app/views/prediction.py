"""Page 2 - Predict churn risk for a single customer."""
from datetime import datetime
import logging

import pandas as pd
import plotly.express as px
import streamlit as st

import components as ui
import data_access as da
from compat import plotly, stretch
from src.config import CATEGORY_OPTIONS as OPT
from src.predict import predict_records, recommendations, risk_band, what_if
from theme import RISK_COLORS, style_fig


def _form() -> dict | None:
    """Render the grouped input form. Returns the customer record on submit."""
    with st.form("customer_form"):
        ui.md('<div class="card-title">Customer profile</div>')
        c1, c2 = st.columns(2)
        gender = c1.selectbox("Gender", OPT["gender"])
        senior = c2.selectbox("Senior citizen", ["No", "Yes"])
        partner = c1.selectbox("Has a partner", OPT["Partner"])
        dependents = c2.selectbox("Has dependents", OPT["Dependents"])

        ui.md('<div class="card-title" style="margin-top:1rem">Account and billing</div>')
        tenure = st.slider("Tenure (months with the company)", 0, 72, 6)
        c1, c2 = st.columns(2)
        contract = c1.selectbox("Contract", OPT["Contract"])
        payment = c2.selectbox("Payment method", OPT["PaymentMethod"])
        paperless = c1.selectbox("Paperless billing", OPT["PaperlessBilling"])
        monthly = c2.number_input("Monthly charges ($)", 15.0, 130.0, 75.0, 0.5)
        auto_total = st.checkbox("Estimate total charges as tenure x monthly charges", value=True)
        total_manual = st.number_input("Total charges ($), used when the estimate is off", 0.0, 10000.0, 0.0, 10.0)

        ui.md('<div class="card-title" style="margin-top:1rem">Services</div>')
        c1, c2 = st.columns(2)
        phone = c1.selectbox("Phone service", OPT["PhoneService"])
        multiple = c2.selectbox("Multiple lines", OPT["MultipleLines"])
        internet = c1.selectbox("Internet service", OPT["InternetService"])
        security = c2.selectbox("Online security", OPT["OnlineSecurity"])
        backup = c1.selectbox("Online backup", OPT["OnlineBackup"])
        protection = c2.selectbox("Device protection", OPT["DeviceProtection"])
        support = c1.selectbox("Tech support", OPT["TechSupport"])
        tv = c2.selectbox("Streaming TV", OPT["StreamingTV"])
        movies = c1.selectbox("Streaming movies", OPT["StreamingMovies"])
        st.caption("Add-on services are treated as 'No' when the customer has no internet (or no phone for multiple lines).")

        submitted = stretch(st.form_submit_button, "Predict churn risk", type="primary")

    if not submitted:
        return None
    return {
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection, "TechSupport": support,
        "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": float(monthly),
        "TotalCharges": float(tenure * monthly) if auto_total else float(total_manual),
    }


def _store(ctx: da.Ctx, record: dict) -> None:
    prob = float(predict_records(ctx.model, [record])[0])
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **record,
        "churn_probability": round(prob, 4), "risk_level": risk_band(prob),
        "predicted_churn": int(prob >= ctx.metrics["threshold"]), "model": ctx.metrics["best_model"],
    }
    st.session_state["last"] = {"record": record, "prob": prob, "entry": entry}
    st.session_state["history"].append(entry)


def _result_panel(ctx: da.Ctx) -> None:
    last = st.session_state.get("last")
    if not last:
        ui.empty_state("Fill in the customer details and select 'Predict churn risk' to see the score, "
                       "the reasons to act and suggested retention offers.")
        return

    prob, record, entry = last["prob"], last["record"], last["entry"]
    level = risk_band(prob)
    likely = prob >= ctx.metrics["threshold"]

    ui.md(f'<div class="card"><div style="display:flex;justify-content:space-between;align-items:center">'
          f'<div class="card-title" style="margin:0">Churn risk</div>{ui.risk_badge(level)}</div></div>')
    plotly(ui.risk_gauge(prob, ctx.dark), key="gauge")
    ui.md(ui.risk_bar(prob))
    verdict = "likely to cancel" if likely else "likely to stay"
    at_stake = record["MonthlyCharges"] * 12
    ui.md(f'<div class="verdict">The model thinks this customer is <b>{verdict}</b>. '
          f'Yearly revenue at stake: <b>{ui.esc(ui.money(at_stake))}</b>.</div>')

    ui.section("Why this score?", "SHAP describes the uncalibrated model's drivers, not causal effects.")
    try:
        from src.explain import explain_record

        drivers = explain_record(record)
        drivers["effect"] = drivers["effect"].map({
            "Raises churn score": "Raises churn risk",
            "Lowers churn score": "Lowers churn risk",
        })
        fig = px.bar(
            drivers, x="impact", y="feature", orientation="h", color="effect",
            color_discrete_map={"Raises churn risk": RISK_COLORS["High"], "Lowers churn risk": RISK_COLORS["Low"]},
        )
        fig.update_yaxes(autorange="reversed", title=None)
        fig.update_xaxes(title="SHAP impact on the uncalibrated model score")
        plotly(style_fig(fig, ctx.dark, height=60 + 42 * len(drivers), legend=False), key="prediction_shap")
        for driver in drivers.itertuples(index=False):
            direction = "increases" if driver.impact >= 0 else "decreases"
            st.caption(f"{driver.feature} {direction} this customer's churn score relative to the model baseline.")
    except Exception:
        logging.getLogger(__name__).exception("Could not calculate local SHAP explanation")
        st.info("A feature explanation is temporarily unavailable. The churn prediction and recommendations are still available.")

    ui.section("Recommended actions")
    ui.recommendation_cards(recommendations(record, prob))

    scenarios = what_if(ctx.model, record, prob)
    if scenarios:
        ui.section("What if we make an offer?", "Estimated change in churn probability. Based on patterns in past data, not a guarantee.")
        sc = pd.DataFrame(scenarios)
        sc["change_pts"] = sc["delta"] * 100
        sc["direction"] = sc["change_pts"].map(lambda v: "Lower risk" if v < 0 else "Higher risk")
        fig = px.bar(sc, x="change_pts", y="scenario", orientation="h", color="direction",
                     color_discrete_map={"Lower risk": RISK_COLORS["Low"], "Higher risk": RISK_COLORS["High"]})
        fig.update_traces(marker_cornerradius=5)
        fig.update_xaxes(title="Change in churn probability (percentage points)")
        fig.update_yaxes(title=None, autorange="reversed")
        plotly(style_fig(fig, ctx.dark, height=60 + 52 * len(sc), legend=False), key="whatif")

    csv = pd.DataFrame([entry]).to_csv(index=False).encode("utf-8")
    stretch(st.download_button, "Download this prediction report (CSV)", csv,
            file_name=f"churn_report_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv", key="dl_single")


def _clear_history() -> None:
    st.session_state["history"] = []
    st.session_state.pop("last", None)


def _history() -> None:
    ui.section("Prediction history", "Predictions made in this browser session.")
    history = st.session_state["history"]
    if not history:
        ui.empty_state("No predictions yet.")
        return
    df = pd.DataFrame(history)
    view = pd.DataFrame({
        "Time": df["timestamp"], "Tenure": df["tenure"].astype(str) + " mo", "Contract": df["Contract"],
        "Internet": df["InternetService"], "Monthly bill": df["MonthlyCharges"].map(lambda v: f"${v:,.2f}"),
        "Churn probability": df["churn_probability"].map(lambda v: f"{v:.0%}"), "Risk": df["risk_level"],
    }).iloc[::-1]
    ui.html_table(view, risk_col="Risk")
    st.write("")
    a, b, _ = st.columns([2, 1, 2])
    with a:
        stretch(st.download_button, "Download full history (CSV)", df.to_csv(index=False).encode("utf-8"),
                file_name="churn_prediction_history.csv", mime="text/csv", key="dl_history")
    with b:
        stretch(st.button, "Clear history", on_click=_clear_history, key="clear_hist")


def render(ctx: da.Ctx) -> None:
    ui.page_header("Customer prediction",
                   "Enter a customer's details to estimate their chance of leaving and get retention ideas.")
    left, right = st.columns([3, 2], gap="large")
    with left:
        record = _form()
        if record is not None:
            _store(ctx, record)
    with right:
        _result_panel(ctx)
    _history()
