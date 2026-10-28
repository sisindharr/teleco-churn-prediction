"""Page 5 - About the project: business case, method, stack, limitations."""
import streamlit as st

import components as ui
import data_access as da

STEPS = [
    ("Clean the data", "Fix the text-typed TotalCharges column, merge 'No internet service' into 'No', convert Churn to 1/0."),
    ("Explore (EDA)", "Chart churn against contract, tenure, price and services to find patterns."),
    ("Engineer features", "Create tenure groups, add-on counts, auto-pay and family flags."),
    ("Preprocess", "Scale numbers and one-hot encode categories inside one reusable scikit-learn pipeline."),
    ("Train three models", "Logistic Regression, Random Forest and XGBoost, with class imbalance handled."),
    ("Evaluate fairly", "5-fold cross-validation picks the winner; a hidden test set reports final scores."),
    ("Deploy", "The pipeline is saved with joblib and served by this Streamlit app."),
]
STACK = ["Python", "Pandas", "NumPy", "Seaborn", "Matplotlib", "Plotly", "Scikit-learn", "XGBoost", "Streamlit", "Git & GitHub"]


def render(ctx: da.Ctx) -> None:
    ui.page_header("About this project", "An end-to-end churn prediction product for a telecom retention team.")

    c1, c2 = st.columns(2)
    with c1:
        ui.card("The business problem",
                "Winning a new customer costs several times more than keeping an existing one. "
                "A telecom company loses roughly one in four customers in this dataset, "
                "but the retention team cannot call everybody. They need to know <b>who</b> is about to leave and <b>why</b>.")
        ui.card("Objectives",
                "1. Predict which customers will churn.<br>2. Explain the main drivers of churn.<br>"
                "3. Turn scores into actions: risk levels, offers and call lists.")
    with c2:
        ui.card("Expected business impact",
                "Focusing offers on the highest-risk customers makes retention budgets work harder. "
                "Use the campaign estimator on the Home page to size the benefit with your own assumptions. "
                "Results depend on how many offers succeed, so treat estimates as scenarios, not promises.")
        ui.card("Dataset",
                f"IBM sample <b>Telco Customer Churn</b> from Kaggle: {ctx.metrics['dataset']['rows']:,} customers, "
                f"21 columns covering demographics, services, contract and billing. "
                f"Churn rate: {ctx.metrics['dataset']['churn_rate']:.1%}.")

    ui.section("How it was built")
    for i, (title, text) in enumerate(STEPS, start=1):
        ui.md(f'<div class="step"><div class="step-n">{i}</div><div><div class="card-title" style="margin:0">{ui.esc(title)}</div>'
              f'<div class="card-text">{ui.esc(text)}</div></div></div>')

    ui.section("Tech stack")
    ui.md("".join(f'<span class="chip">{ui.esc(s)}</span>' for s in STACK))

    ui.section("Limitations and responsible use")
    ui.card("Please read",
            "The data is a public sample, not a real company's customers. The model finds <b>associations</b>, not proven causes, "
            "so the 'what if' tool is a guide for testing offers, not a forecast. "
            "Scores should support human decisions, not replace them, and should be monitored for drift and fairness over time.")

    ui.section("Customer value assumptions")
    ui.card("How the retention ranking works",
            "The Home and Batch Scoring pages estimate customer lifetime value as "
            "<b>monthly charges × gross margin × expected remaining months</b>. "
            "Revenue at risk is <b>churn probability × estimated lifetime value</b>. "
            "The defaults are a 30% gross margin and 24 remaining months; both can be adjusted with sliders. "
            "These are planning assumptions, not an accounting valuation.")

    ui.section("Author")
    ui.card("Your Name",
            "Data Scientist &middot; <a href='https://github.com/your-username' target='_blank'>GitHub</a> &middot; "
            "<a href='https://www.linkedin.com/in/your-profile' target='_blank'>LinkedIn</a><br>"
            f"Environment: Python {ctx.metrics['versions']['python']}, scikit-learn {ctx.metrics['versions']['scikit_learn']}, "
            f"XGBoost {ctx.metrics['versions']['xgboost']}.")
