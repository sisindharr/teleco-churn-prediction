"""Page 4 - Model performance: comparison, ROC, confusion matrix, importance."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import logging

import components as ui
import data_access as da
from compat import plotly
from theme import RISK_COLORS, style_fig

METRIC_LABELS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1-score", "roc_auc": "ROC-AUC"}


def render(ctx: da.Ctx) -> None:
    m = ctx.metrics
    models = m["models"]
    best = m["best_model"]
    ui.page_header("Model performance",
                   "Three models were trained on the same data and graded on customers they had never seen.")

    ui.callout(f"Selected model: {best}",
               "Chosen by cross-validated ROC-AUC on the training data, so the test results below are an honest, unbiased estimate. "
               f"Decision threshold: {m['threshold']:.0%}. Dataset: {m['dataset']['rows']:,} customers, "
               f"{m['dataset']['train_size']:,} for training and {m['dataset']['test_size']:,} for testing.")

    # ---- comparison table -------------------------------------------------------
    ui.section("Model comparison", "Higher is better for every column.")
    rows = []
    for name, res in models.items():
        rows.append({"Model": name, **{METRIC_LABELS[k]: f"{v:.3f}" for k, v in res["metrics"].items()},
                     "CV ROC-AUC": f"{res['cv_roc_auc_mean']:.3f} ± {res['cv_roc_auc_std']:.3f}"})
    table = pd.DataFrame(rows)
    ui.html_table(table, highlight_row=list(models).index(best))

    calibration = m.get("calibration")
    if calibration:
        ui.section("Calibration", "The chosen calibration method is fitted using training data only.")
        ui.callout(
            "What calibration means",
            "Calibration makes predicted percentages better match real-world frequencies. "
            "For example, among customers scored near 30%, about 30% should churn. "
            "The method is selected on a separate training-only split; this chart is checked on the untouched test set.",
        )
        ui.md(
            f'<div class="card-text">Selected method: <b>{ui.esc(calibration["method"].title())}</b>. '
            f'Brier score before: <b>{calibration["brier_before"]:.3f}</b>; '
            f'after: <b>{calibration["brier_after"]:.3f}</b> (lower is better). '
            "Risk bands remain Low &lt; 30%, Medium 30-60%, High &ge; 60%; after calibration, "
            "these thresholds can be read as approximate probabilities, but should be tuned to retention capacity and costs.</div>"
        )
        before = calibration["reliability_before"]
        after = calibration["reliability_after"]
        fig = go.Figure()
        fig.add_scatter(x=before["mean_predicted_probability"], y=before["fraction_of_positives"],
                        mode="lines+markers", name="Before calibration")
        fig.add_scatter(x=after["mean_predicted_probability"], y=after["fraction_of_positives"],
                        mode="lines+markers", name="After calibration")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                        line=dict(dash="dash", color="#94A3B8"))
        fig.update_xaxes(title="Mean predicted probability", range=[0, 1], tickformat=".0%")
        fig.update_yaxes(title="Observed churn rate", range=[0, 1], tickformat=".0%")
        plotly(style_fig(fig, ctx.dark, title="Reliability curve"), key="perf_calibration")

    c1, c2 = st.columns(2)
    with c1:
        long = pd.DataFrame([{"Model": n, "Metric": METRIC_LABELS[k], "Score": v}
                             for n, r in models.items() for k, v in r["metrics"].items()])
        fig = px.bar(long, x="Metric", y="Score", color="Model", barmode="group")
        fig.update_traces(marker_cornerradius=4)
        fig.update_yaxes(range=[0, 1], title=None)
        fig.update_xaxes(title=None)
        plotly(style_fig(fig, ctx.dark, height=360, title="Metrics side by side"), key="perf_bars")
    with c2:
        fig = go.Figure()
        for name, res in models.items():
            auc = res["metrics"]["roc_auc"]
            fig.add_scatter(x=res["roc_curve"]["fpr"], y=res["roc_curve"]["tpr"], mode="lines",
                            name=f"{name} (AUC {auc:.3f})", line=dict(width=3 if name == best else 2))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guessing", line=dict(dash="dash", color="#94A3B8"))
        fig.update_xaxes(title="False positive rate")
        fig.update_yaxes(title="True positive rate (recall)")
        plotly(style_fig(fig, ctx.dark, height=360, title="ROC curves"), key="perf_roc")

    # ---- confusion matrix + importance -----------------------------------------
    names = list(models)
    choice = st.radio("Inspect model", names, index=names.index(best), horizontal=True, key="perf_model")
    res = models[choice]
    c1, c2 = st.columns(2)
    with c1:
        (tn, fp), (fn, tp) = res["confusion_matrix"]
        fig = px.imshow(res["confusion_matrix"], text_auto=True, color_continuous_scale="Purples", aspect="auto",
                        x=["Predicted: stay", "Predicted: churn"], y=["Actually stayed", "Actually churned"])
        fig.update_layout(coloraxis_showscale=False)
        plotly(style_fig(fig, ctx.dark, height=340, legend=False, title=f"Confusion matrix: {choice}"), key="perf_cm")
        caught = tp / (tp + fn) if (tp + fn) else 0
        ui.md(f'<div class="card-text">Of <b>{tp + fn}</b> customers who really left, the model caught <b>{tp}</b> '
              f'({caught:.0%}) and missed <b>{fn}</b>. It also flagged <b>{fp}</b> loyal customers by mistake.</div>')
    with c2:
        imp = pd.DataFrame(res["feature_importance"]).head(15)
        if "coefficient" in imp:
            imp["effect"] = imp["coefficient"].map(lambda v: "Raises churn risk" if v > 0 else "Lowers churn risk")
            fig = px.bar(imp, x="importance", y="feature", orientation="h", color="effect",
                         color_discrete_map={"Raises churn risk": RISK_COLORS["High"], "Lowers churn risk": RISK_COLORS["Low"]})
            legend = True
        else:
            fig = px.bar(imp, x="importance", y="feature", orientation="h", color_discrete_sequence=["#6366F1"])
            legend = False
        fig.update_traces(marker_cornerradius=4)
        fig.update_yaxes(autorange="reversed", title=None)
        fig.update_xaxes(title="Importance")
        plotly(style_fig(fig, ctx.dark, height=460, legend=legend, title=f"Top drivers of churn: {choice}"), key="perf_imp")

    ui.section("Global SHAP summary", "Average feature impact across up to 500 customers.")
    ui.md('<div class="card-text">SHAP explains the uncalibrated model drivers. It describes patterns learned by the model, not causes.</div>')
    try:
        from src.explain import global_importance

        importance = global_importance(ctx.data).head(20).sort_values("importance")
        fig = px.bar(importance, x="importance", y="feature", orientation="h",
                     color_discrete_sequence=[RISK_COLORS["High"]])
        fig.update_yaxes(title=None)
        fig.update_xaxes(title="Mean absolute SHAP impact")
        plotly(style_fig(fig, ctx.dark, height=max(360, 24 * len(importance)),
                         legend=False, title="Global feature impact"), key="perf_shap")
    except Exception:
        logging.getLogger(__name__).exception("Could not calculate global SHAP summary")
        st.info("The SHAP summary is temporarily unavailable. Model metrics and other performance charts are unaffected.")

    # ---- beginner guide ---------------------------------------------------------
    ui.section("How to read these numbers")
    with st.expander("Plain-English guide to the metrics", expanded=False):
        st.markdown(
            "- **Accuracy**: share of all predictions that were right. Can be misleading here because only about 1 in 4 customers churn.\n"
            "- **Precision**: when the model says *will churn*, how often is it right? Low precision wastes retention offers.\n"
            "- **Recall**: of the customers who really left, how many did the model catch? Low recall means lost revenue.\n"
            "- **F1-score**: one number that balances precision and recall.\n"
            "- **ROC-AUC**: how well the model ranks risky customers above safe ones. 0.5 is a coin flip, 1.0 is perfect.\n"
            "- **Confusion matrix**: the full count of right and wrong calls (top-left: correctly kept, bottom-right: correctly caught churners).\n"
            "- **Feature importance**: which customer attributes the model leans on most. It shows association, not proof of cause.")
    ui.callout("Business trade-off",
               "Missing a churner (false negative) costs months of revenue, while a wasted offer (false positive) costs a discount. "
               "That is why models here are tuned to catch more churners, even if some offers go to customers who would have stayed.")
