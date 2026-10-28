"""ChurnIQ - Streamlit entry point.

Run locally:  streamlit run app/main.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="ChurnIQ | Customer Retention Analytics", page_icon="📉",
                   layout="wide", initial_sidebar_state="expanded")

import components as ui  # noqa: E402
import data_access as da  # noqa: E402
import theme  # noqa: E402
from views import about, batch, home, insights, performance, prediction  # noqa: E402

PAGES = {
    "Home Dashboard": ("🏠", home.render),
    "Customer Prediction": ("🎯", prediction.render),
    "Batch Scoring": ("📥", batch.render),
    "Data Insights": ("📊", insights.render),
    "Model Performance": ("🧪", performance.render),
    "About Project": ("ℹ️", about.render),
}

# ---- session state ---------------------------------------------------------
st.session_state.setdefault("nav", "Home Dashboard")
st.session_state.setdefault("history", [])

ready = da.artifacts_ready()

# ---- sidebar ---------------------------------------------------------------
with st.sidebar:
    logo = ROOT / "assets" / "logo.png"
    if logo.exists():  # drop your own logo at assets/logo.png
        st.image(str(logo), width=150)
    else:  # logo placeholder
        ui.md('<div class="brand"><div class="brand-logo">C</div><div><div class="brand-name">ChurnIQ</div>'
              '<div class="brand-tag">Customer retention analytics</div></div></div>')

    st.radio("Navigation", list(PAGES), key="nav", label_visibility="collapsed",
             format_func=lambda p: f"{PAGES[p][0]}  {p}")
    st.write("")
    st.toggle("Dark mode", key="dark_mode")

    if ready:
        meta = da.load_metrics()
        best = meta["models"][meta["best_model"]]["metrics"]
        ui.md(f'<div class="side-card"><b>Live model</b><br>{ui.esc(meta["best_model"])}<br>'
              f'ROC-AUC {best["roc_auc"]:.3f} &middot; Recall {best["recall"]:.0%}<br>'
              f'Trained {ui.esc(meta["trained_at"][:10])}</div>')
        import sklearn
        if meta["versions"]["scikit_learn"] != sklearn.__version__:
            st.warning(f"Model was trained with scikit-learn {meta['versions']['scikit_learn']}, "
                       f"but {sklearn.__version__} is installed. Retrain with `python -m src.train`.")

dark = bool(st.session_state.get("dark_mode", False))
ui.md(theme.css(dark))

# ---- first-run guard -------------------------------------------------------
if not ready:
    ui.page_header("Almost there", "The trained model and processed data were not found.")
    ui.callout("Run the training pipeline first",
               "1) Download the Kaggle 'Telco Customer Churn' CSV into data/raw/  "
               "2) Run: python -m src.train  3) Refresh this page.")
    st.stop()

ctx = da.Ctx(model=da.load_model(), metrics=da.load_metrics(), data=da.load_data(), dark=dark)
PAGES[st.session_state["nav"]][1](ctx)
