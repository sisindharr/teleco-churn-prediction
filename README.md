# ChurnIQ: Telco Customer Churn Prediction

An end-to-end data science project: from raw Kaggle data to a deployed, SaaS-style Streamlit analytics app that predicts which telecom customers are about to leave and recommends what to do about it.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B) ![scikit--learn](https://img.shields.io/badge/scikit--learn-pipeline-F7931E) ![XGBoost](https://img.shields.io/badge/XGBoost-model-1f77b4) ![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF)

> **Live demo:** _add your Streamlit Cloud link here_  |  **Screenshots:** _add to `docs/screenshots/`_

## Why this project?

Winning a new customer costs several times more than keeping one. A telecom firm in this dataset loses about **1 in 4 customers**, but the retention team cannot call everyone. ChurnIQ answers three questions:

1. **Who** is likely to leave? (probability + Low / Medium / High risk)
2. **Why**? (drivers such as contract type, tenure, services, billing)
3. **What should we do?** (rule-based next-best actions and "what-if" offer simulation)

**Expected business impact:** target retention offers at the highest-risk customers instead of everyone, so budget goes further and fewer valuable customers leave. The in-app campaign estimator lets you size the benefit with your own assumptions.

## Features

| Area | What you get |
|---|---|
| Data science | Cleaning, EDA (Seaborn/Matplotlib), feature engineering, one scikit-learn `Pipeline`, 3-model comparison, cross-validation, training-only probability calibration |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, Brier score, reliability curve, confusion matrix, ROC curves, feature importance, SHAP |
| App | 6 pages, dark/light mode, custom CSS, KPI cards, risk gauge, SHAP explanations, CLV-based customer ranking, batch CSV scoring, Plotly charts, prediction history, CSV downloads, mobile-friendly layout |
| Engineering | Modular `src/` package, tests on synthetic data, Dockerfile, GitHub Actions CI |

**App pages:** Home Dashboard, Customer Prediction, Data Insights, Model Performance, Batch Scoring, About Project.

## Project structure

```
telco-churn-prediction/
├── app/
│   ├── main.py              # Streamlit entry point + sidebar navigation
│   ├── theme.py             # design tokens, global CSS (light/dark), Plotly styling
│   ├── components.py        # KPI cards, badges, tables, risk gauge
│   ├── data_access.py       # cached loading + scoring
│   ├── compat.py            # Streamlit version helpers
│   └── views/               # home, prediction, insights, performance, batch, about
├── src/
│   ├── config.py            # paths, column groups, thresholds
│   ├── data_cleaning.py     # step 1: cleaning
│   ├── features.py          # step 2: feature engineering
│   ├── eda.py               # step 2b: EDA figures
│   ├── train.py             # step 3: training, comparison, evaluation, saving
│   ├── predict.py           # risk bands, what-if, recommendations
│   ├── explain.py           # SHAP explanations for the uncalibrated base model
│   ├── value.py             # lifetime value and retention-priority calculations
│   └── batch.py             # in-memory validation and batch scoring
├── notebooks/01_churn_walkthrough.ipynb   # beginner-friendly step-by-step
├── tests/                   # pytest (uses synthetic data)
├── docs/                    # project guide, deployment, resume text, interview Q&A
├── data/raw/                # put the Kaggle CSV here (not committed)
├── data/processed/          # cleaned data (committed, used by the app)
├── models/                  # calibrated + base pipelines and metrics.json (committed)
├── .streamlit/config.toml
├── .github/workflows/ci.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Installation guide

**Prerequisites:** Python 3.11 or newer, Git.

```bash
# 1. Clone
git clone https://github.com/<your-username>/telco-churn-prediction.git
cd telco-churn-prediction

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Get the data (free Kaggle account needed)
#    https://www.kaggle.com/datasets/blastchar/telco-customer-churn
#    Save the file as: data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv

# 5. (Optional) Generate EDA charts into reports/figures/
python -m src.eda

# 6. Train and compare the models  (add --tune for hyper-parameter search)
python -m src.train

# 7. Launch the app
streamlit run app/main.py
```

Step 6 creates `models/churn_pipeline.joblib` (calibrated), `models/churn_base_pipeline.joblib` (SHAP), `models/metrics.json` and `data/processed/churn_processed.csv`. The app reads those files.

Run the tests (no Kaggle download needed): `pip install -r requirements-dev.txt && pytest -q`

## Results

These scores are from the current real Kaggle-data run. The model comparison metrics below use each base model; calibration is evaluated separately on the held-out test set. Re-run `python -m src.train` after changing the data or code.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.734 | 0.499 | 0.781 | 0.609 | 0.843 |
| Random Forest | 0.761 | 0.537 | 0.711 | 0.612 | 0.841 |
| XGBoost | 0.742 | 0.510 | 0.775 | 0.615 | 0.843 |

The winning model is chosen by **cross-validated ROC-AUC on the training set**, so the test set stays an unbiased final exam.

The winning model is calibrated by comparing sigmoid and isotonic calibration on a separate slice of training data. In this run, isotonic was selected: test Brier score improved from **0.166** to **0.139** (lower is better). The reliability curves and method are in `models/metrics.json`. The app uses the calibrated pipeline for probabilities; SHAP explains the matching uncalibrated base pipeline.

## Deployment

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for Streamlit Community Cloud, Docker, and CI instructions. Short version:

1. Train locally, then commit `models/` and `data/processed/`.
2. Push to GitHub.
3. On [share.streamlit.io](https://share.streamlit.io) choose the repo, branch `main`, main file `app/main.py`.

## Documentation

- [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md): business case and every step explained for beginners
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md): deployment guide
- [`docs/RESUME.md`](docs/RESUME.md): resume and LinkedIn text
- [`docs/INTERVIEW_QA.md`](docs/INTERVIEW_QA.md): 15 interview questions with answers

## Limitations

- Public sample data, not a real company's customers.
- Feature importance and the "what-if" tool show **associations**, not proven causes. Validate offers with an A/B test.
- Calibration makes probabilities more useful as frequency estimates, but does not guarantee future calibration after customer behavior changes. Review risk thresholds against actual campaign capacity and costs.
- CLV and revenue-at-risk are planning estimates using editable assumptions (30% gross margin and 24 remaining months by default), not accounting forecasts.
- Prediction history lives in the browser session only.

## Roadmap

Uplift modelling, probability drift monitoring, a database for prediction history.

## License

MIT, see [`LICENSE`](LICENSE).
