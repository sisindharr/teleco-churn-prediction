"""Page 3 - Data insights: interactive exploration of churn drivers."""
import plotly.express as px
import streamlit as st

import components as ui
import data_access as da
from compat import plotly
from src.features import TENURE_LABELS
from theme import CHURN_COLORS, style_fig


def _rate_bar(df, col, title, ctx, key, order=None):
    t = da.churn_rate_by(df, col)
    if order:
        t[col] = t[col].astype("category").cat.set_categories(order, ordered=True)
        t = t.sort_values(col)
    fig = px.bar(t, x=col, y="rate", color_discrete_sequence=["#6366F1"], hover_data={"customers": True})
    fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside", marker_cornerradius=6)
    fig.update_yaxes(title=None, ticksuffix="%", range=[0, max(t["rate"].max() * 1.25, 5)])
    fig.update_xaxes(title=None)
    plotly(style_fig(fig, ctx.dark, height=300, legend=False, title=title), key=key)


def _findings(df) -> list[tuple[str, str]]:
    """Plain-English findings computed from the (filtered) data."""
    out = []
    for col, label in [("Contract", "contract"), ("InternetService", "internet service"), ("PaymentMethod", "payment method")]:
        t = da.churn_rate_by(df, col).sort_values("rate")
        if len(t) < 2:
            continue
        lo, hi = t.iloc[0], t.iloc[-1]
        out.append((f"Highest churn by {label}",
                    f"'{hi[col]}' customers churn at {hi['rate']:.1f}%, compared with {lo['rate']:.1f}% for '{lo[col]}'."))
    return out


def render(ctx: da.Ctx) -> None:
    df = ctx.data
    ui.page_header("Data insights", "Explore which customers leave and why. Use the filters to focus on a segment.")

    with st.expander("Filters", expanded=False):
        a, b = st.columns(2)
        contracts = a.multiselect("Contract", sorted(df["Contract"].unique()), default=sorted(df["Contract"].unique()))
        internet = b.multiselect("Internet service", sorted(df["InternetService"].unique()), default=sorted(df["InternetService"].unique()))
        tenure = st.slider("Tenure (months)", 0, int(df["tenure"].max()), (0, int(df["tenure"].max())))

    f = df[df["Contract"].isin(contracts) & df["InternetService"].isin(internet) & df["tenure"].between(*tenure)]
    if f.empty:
        ui.empty_state("No customers match these filters. Widen the selection to see charts.")
        return

    k = st.columns(4)
    with k[0]:
        ui.kpi_card("Customers in view", f"{len(f):,}", f"{len(f) / len(df):.0%} of all customers", "👥")
    with k[1]:
        ui.kpi_card("Churn rate", f"{f['Churn'].mean():.1%}", f"overall: {df['Churn'].mean():.1%}", "📉", "#F43F5E")
    with k[2]:
        ui.kpi_card("Average monthly bill", ui.money(f["MonthlyCharges"].mean()), "per customer", "💳", "#22D3EE")
    with k[3]:
        ui.kpi_card("Average tenure", f"{f['tenure'].mean():.0f} mo", "time with the company", "⏳", "#F59E0B")

    for title, text in _findings(f):
        ui.callout(title, text)

    tab_over, tab_people, tab_services, tab_billing = st.tabs(["Overview", "Customer profile", "Services", "Billing"])

    with tab_over:
        c1, c2 = st.columns(2)
        with c1:
            counts = f["ChurnLabel"].value_counts().reset_index()
            counts.columns = ["status", "customers"]
            fig = px.pie(counts, names="status", values="customers", hole=0.6, color="status", color_discrete_map=CHURN_COLORS)
            plotly(style_fig(fig, ctx.dark, title="Retained vs churned"), key="ins_pie")
        with c2:
            fig = px.histogram(f, x="tenure", color="ChurnLabel", nbins=36, barmode="overlay", opacity=0.75,
                               color_discrete_map=CHURN_COLORS)
            fig.update_xaxes(title="Tenure (months)")
            fig.update_yaxes(title="Customers")
            plotly(style_fig(fig, ctx.dark, title="Most churn happens early"), key="ins_hist")
        _rate_bar(f, "tenure_group", "Churn rate by time with the company", ctx, "ins_tg", TENURE_LABELS)

    with tab_people:
        c1, c2 = st.columns(2)
        with c1:
            _rate_bar(f, "gender", "Churn rate by gender", ctx, "ins_gender")
            _rate_bar(f, "Partner", "Churn rate by partner status", ctx, "ins_partner")
        with c2:
            g = f.assign(SeniorCitizen=f["SeniorCitizen"].map({0: "Not senior", 1: "Senior"}))
            _rate_bar(g, "SeniorCitizen", "Churn rate by age group", ctx, "ins_senior")
            _rate_bar(f, "Dependents", "Churn rate by dependents", ctx, "ins_dep")

    with tab_services:
        c1, c2 = st.columns(2)
        with c1:
            _rate_bar(f, "InternetService", "Churn rate by internet service", ctx, "ins_internet")
            _rate_bar(f, "OnlineSecurity", "Churn rate by online security", ctx, "ins_sec")
        with c2:
            _rate_bar(f, "TechSupport", "Churn rate by tech support", ctx, "ins_support")
            _rate_bar(f, "StreamingTV", "Churn rate by streaming TV", ctx, "ins_tv")
        pivot = f.pivot_table(index="InternetService", columns="Contract", values="Churn", aggfunc="mean") * 100
        fig = px.imshow(pivot, text_auto=".1f", aspect="auto", color_continuous_scale="RdYlGn_r")
        fig.update_layout(coloraxis_showscale=False)
        plotly(style_fig(fig, ctx.dark, height=260, legend=False, title="Churn rate (%) by internet service and contract"), key="ins_heat")

    with tab_billing:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(f, x="ChurnLabel", y="MonthlyCharges", color="ChurnLabel", color_discrete_map=CHURN_COLORS)
            fig.update_xaxes(title=None)
            fig.update_yaxes(title="Monthly charges ($)")
            plotly(style_fig(fig, ctx.dark, legend=False, title="Churners pay more per month"), key="ins_box")
        with c2:
            _rate_bar(f, "PaymentMethod", "Churn rate by payment method", ctx, "ins_pay")
        sample = f.sample(min(len(f), 1500), random_state=1)
        fig = px.scatter(sample, x="tenure", y="MonthlyCharges", color="ChurnLabel", opacity=0.6, color_discrete_map=CHURN_COLORS)
        fig.update_xaxes(title="Tenure (months)")
        fig.update_yaxes(title="Monthly charges ($)")
        plotly(style_fig(fig, ctx.dark, height=380, title="Tenure vs monthly charges (sample of customers)"), key="ins_scatter")
        num = f[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "num_addon_services", "has_autopay", "has_family", "Churn"]]
        fig = px.imshow(num.corr(), text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto")
        fig.update_layout(coloraxis_showscale=False)
        plotly(style_fig(fig, ctx.dark, height=420, legend=False, title="Correlation between numeric features and churn"), key="ins_corr")
