"""Page 1 - Home dashboard: KPIs, risk segments and the call list."""
import pandas as pd
import plotly.express as px
import streamlit as st

import components as ui
import data_access as da
import theme
from compat import plotly, stretch
from src.features import TENURE_LABELS
from src.value import add_customer_value
from theme import RISK_COLORS, style_fig


def render(ctx: da.Ctx) -> None:
    df = ctx.data
    scored = da.scored_data(ctx)
    active = scored[scored["Churn"] == 0]
    best = ctx.metrics["models"][ctx.metrics["best_model"]]["metrics"]

    ui.page_header("Retention dashboard",
                   "See who is likely to leave, what it costs, and who to contact first.")

    with st.expander("Customer lifetime value assumptions", expanded=False):
        a, b = st.columns(2)
        expected_months = a.slider("Expected remaining months", 1, 60, 24)
        gross_margin = b.slider("Gross margin", 5, 100, 30, 5, format="%d%%") / 100
    active = add_customer_value(
        active, gross_margin=gross_margin, expected_remaining_months=expected_months)
    high_active = active[active["risk"] == "High"]

    # ---- KPI cards ----------------------------------------------------------
    churned = df[df["Churn"] == 1]
    row1 = st.columns(4)
    with row1[0]:
        ui.kpi_card("Total customers", f"{len(df):,}", f"{len(active):,} still active", "👥")
    with row1[1]:
        ui.kpi_card("Churn rate", f"{df['Churn'].mean():.1%}", f"{len(churned):,} customers left", "📉", "#F43F5E")
    with row1[2]:
        ui.kpi_card("Average monthly bill", ui.money(df["MonthlyCharges"].mean()), "per customer", "💳", "#22D3EE")
    with row1[3]:
        ui.kpi_card("Monthly revenue lost", ui.money(churned["MonthlyCharges"].sum()), "from churned customers", "💸", "#F59E0B")
    row2 = st.columns(5)
    with row2[0]:
        ui.kpi_card("Average tenure", f"{df['tenure'].mean():.0f} mo", "time with the company", "⏳")
    with row2[1]:
        ui.kpi_card("Active at high risk", f"{len(high_active):,}", f"{ui.money(high_active['MonthlyCharges'].sum())} monthly revenue", "🚨", "#EF4444")
    with row2[2]:
        ui.kpi_card("Model ROC-AUC", f"{best['roc_auc']:.3f}", ctx.metrics["best_model"], "🎯", "#10B981")
    with row2[3]:
        ui.kpi_card("Churners caught", f"{best['recall']:.0%}", "model recall on unseen data", "🔎", "#10B981")
    with row2[4]:
        ui.kpi_card("Total revenue at risk", ui.money(active["revenue_at_risk"].sum()),
                    f"CLV at {gross_margin:.0%} margin over {expected_months} months", "💰", "#F59E0B")

    # ---- Charts ------------------------------------------------------------
    ui.section("Where churn happens", "Share of customers who left, by segment.")
    c1, c2 = st.columns(2)
    with c1:
        t = da.churn_rate_by(df, "Contract")
        fig = px.bar(t, x="Contract", y="rate", text="rate", color_discrete_sequence=["#6366F1"])
        fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside", marker_cornerradius=6)
        fig.update_yaxes(title=None, ticksuffix="%", range=[0, max(t["rate"]) * 1.25])
        fig.update_xaxes(title=None)
        plotly(style_fig(fig, ctx.dark, legend=False, title="Churn rate by contract type"), key="home_contract")
    with c2:
        t = da.churn_rate_by(df, "tenure_group")
        t["tenure_group"] = pd.Categorical(t["tenure_group"], TENURE_LABELS, ordered=True)
        t = t.sort_values("tenure_group")
        fig = px.bar(t, x="tenure_group", y="rate", text="rate", color_discrete_sequence=["#22D3EE"])
        fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside", marker_cornerradius=6)
        fig.update_yaxes(title=None, ticksuffix="%", range=[0, max(t["rate"]) * 1.25])
        fig.update_xaxes(title=None)
        plotly(style_fig(fig, ctx.dark, legend=False, title="Churn rate by time with the company"), key="home_tenure")

    ui.section("Risk of today's active customers", "Scores come from the live model. Low < 30%, Medium 30-60%, High >= 60%.")
    c1, c2 = st.columns(2)
    order = ["Low", "Medium", "High"]
    with c1:
        counts = active["risk"].value_counts().reindex(order).fillna(0).reset_index()
        counts.columns = ["risk", "customers"]
        fig = px.pie(counts, names="risk", values="customers", hole=0.62, color="risk", color_discrete_map=RISK_COLORS)
        fig.update_traces(textinfo="percent", sort=False)
        plotly(style_fig(fig, ctx.dark, title="Active customers by risk level"), key="home_risk_pie")
    with c2:
        rev = active.groupby("risk")["MonthlyCharges"].sum().reindex(order).fillna(0).reset_index()
        fig = px.bar(rev, x="risk", y="MonthlyCharges", color="risk", color_discrete_map=RISK_COLORS, text="MonthlyCharges")
        fig.update_traces(texttemplate="$%{y:,.0f}", textposition="outside", marker_cornerradius=6)
        fig.update_yaxes(title=None, tickprefix="$", range=[0, max(rev["MonthlyCharges"].max(), 1) * 1.25])
        fig.update_xaxes(title=None)
        plotly(style_fig(fig, ctx.dark, legend=False, title="Monthly revenue by risk level"), key="home_risk_rev")

    # ---- Call list ----------------------------------------------------------
    ui.section("Contact these customers first",
               "Ranked by expected revenue at risk: churn probability multiplied by customer lifetime value.")
    top = active.sort_values("priority_rank").head(20)
    table = pd.DataFrame({
        "Customer": top["customerID"],
        "Churn probability": top["churn_prob"].map(lambda v: f"{v:.0%}"),
        "Risk": top["risk"],
        "Monthly bill": top["MonthlyCharges"].map(lambda v: f"${v:,.2f}"),
        "Revenue at risk": top["revenue_at_risk"].map(ui.money),
    })
    ui.html_table(table, risk_col="Risk")
    st.caption("Note: the model was trained on 80% of these customers, so scores here are for demonstration. "
               "In production you would score a fresh customer list.")

    # ---- ROI estimator ----------------------------------------------------
    ui.section("Retention campaign estimator", "What could a retention campaign on the high-risk group be worth?")
    with st.expander("Open the estimator", expanded=False):
        a, b, c = st.columns(3)
        success = a.slider("Offers that keep the customer", 5, 60, 25, format="%d%%") / 100
        cost = b.number_input("Cost per offer ($)", 0.0, 500.0, 25.0, 5.0)
        months = c.slider("Months of revenue kept", 3, 36, 12)
        n = len(high_active)
        saved = n * success
        protected = saved * (high_active["MonthlyCharges"].mean() if n else 0) * months
        spend = n * cost
        k = st.columns(4)
        with k[0]:
            ui.kpi_card("Customers targeted", f"{n:,}", "high-risk and active", "🎯")
        with k[1]:
            ui.kpi_card("Expected to stay", f"{saved:,.0f}", f"at {success:.0%} success", "🤝", "#10B981")
        with k[2]:
            ui.kpi_card("Revenue protected", ui.money(protected), f"over {months} months", "🛡️", "#10B981")
        with k[3]:
            ui.kpi_card("Net benefit", ui.money(protected - spend), f"after {ui.money(spend)} campaign cost", "💰",
                        "#10B981" if protected >= spend else "#EF4444")

    # ---- Quick actions ------------------------------------------------------
    ui.section("Next steps")
    q = st.columns(3)
    with q[0]:
        stretch(st.button, "Score a customer", type="primary", on_click=ui.goto, args=("Customer Prediction",), key="go_pred")
    with q[1]:
        stretch(st.button, "Explore the data", on_click=ui.goto, args=("Data Insights",), key="go_ins")
    with q[2]:
        stretch(st.button, "Compare the models", on_click=ui.goto, args=("Model Performance",), key="go_perf")
