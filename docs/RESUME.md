# Resume and profile text

## Resume project entry

**ChurnIQ: Customer Churn Prediction and Retention Analytics App** | Python, Pandas, Scikit-learn, XGBoost, Plotly, Streamlit, Docker, GitHub Actions
- Built an end-to-end churn prediction product on 7,000+ telecom customers: data cleaning, EDA, feature engineering and a leakage-safe scikit-learn pipeline.
- Compared Logistic Regression, Random Forest and XGBoost using 5-fold cross-validation, handling a roughly 27% churn class imbalance with class weighting; selected the model by ROC-AUC (**[X.XX]**) and reported precision, recall (**[XX]%**), F1 and confusion matrices on a held-out test set.
- Translated model output into business actions: Low/Medium/High risk bands, rule-based retention recommendations, a "what-if" offer simulator and a campaign ROI estimator.
- Designed and deployed a multi-page Streamlit dashboard (dark/light themes, custom CSS, Plotly charts, prediction history, CSV export) on Streamlit Cloud, with Docker packaging and a GitHub Actions CI pipeline running automated tests.

_Replace the bracketed values with your own results from `python -m src.train`._

## One-line version
End-to-end customer churn prediction app (scikit-learn, XGBoost, Streamlit) with explainable drivers, risk scoring and retention recommendations, deployed with CI/CD.

## LinkedIn "Projects" description
I built ChurnIQ, a full-stack data science project that predicts which telecom customers are likely to cancel and recommends how to keep them. I cleaned and explored the Kaggle Telco dataset, engineered business-driven features, and compared three models with cross-validation. I turned the best model into a SaaS-style Streamlit app with risk gauges, interactive Plotly insights, a what-if offer simulator and downloadable reports, then packaged it with Docker and automated testing on GitHub Actions. Skills: data analysis, machine learning, business storytelling, deployment.

## 30-second interview pitch
"Telecom companies lose about a quarter of customers in this dataset, and retention teams can't call everyone. I built a model that scores every customer's churn risk, explains the drivers, and recommends an action. I compared three algorithms with cross-validation, focused on recall because missing a churner is more expensive than a wasted offer, and shipped the result as a Streamlit app that a retention manager can actually use."
