"""Synthetic Telco-like data used ONLY for automated tests / CI.

It mimics the schema of the Kaggle file (including the blank TotalCharges quirk) so the
pipeline can be tested without downloading the real dataset. It is NOT real data;
never report results from it.
"""
import numpy as np
import pandas as pd


def make_synthetic_telco(n: int = 1500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pick = lambda opts, p=None: rng.choice(opts, n, p=p)  # noqa: E731

    tenure = np.clip(rng.gamma(2.0, 18, n).round().astype(int), 0, 72)
    contract = pick(["Month-to-month", "One year", "Two year"], [0.55, 0.21, 0.24])
    internet = pick(["DSL", "Fiber optic", "No"], [0.34, 0.44, 0.22])
    phone = pick(["Yes", "No"], [0.9, 0.1])
    payment = pick(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    def addon():
        vals = pick(["Yes", "No"])
        return np.where(internet == "No", "No internet service", vals)

    online_sec, backup, protection, support, tv, movies = (addon() for _ in range(6))
    multiple = np.where(phone == "No", "No phone service", pick(["Yes", "No"]))

    base = np.select([internet == "DSL", internet == "Fiber optic"], [45, 80], default=20)
    n_addons = sum((a == "Yes").astype(int) for a in (online_sec, backup, protection, support, tv, movies))
    monthly = np.round(base + 4 * n_addons + rng.normal(0, 4, n), 2)
    total = np.round(monthly * tenure + rng.normal(0, 20, n).clip(-50, 50) * (tenure > 0), 2).clip(0)

    logit = (-0.8 + 1.3 * (contract == "Month-to-month") - 0.9 * (contract == "Two year")
             - 0.035 * tenure + 0.7 * (internet == "Fiber optic") - 0.7 * (support == "Yes")
             + 0.5 * (payment == "Electronic check") + 0.01 * (monthly - 60))
    churn = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")

    df = pd.DataFrame({
        "customerID": [f"{i:04d}-SYNTH" for i in range(n)],
        "gender": pick(["Female", "Male"]),
        "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": pick(["Yes", "No"]), "Dependents": pick(["Yes", "No"], [0.3, 0.7]),
        "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple,
        "InternetService": internet, "OnlineSecurity": online_sec, "OnlineBackup": backup,
        "DeviceProtection": protection, "TechSupport": support, "StreamingTV": tv,
        "StreamingMovies": movies, "Contract": contract,
        "PaperlessBilling": pick(["Yes", "No"]), "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total.astype(str), "Churn": churn,
    })
    # Reproduce the real dataset's quirk: blank TotalCharges for brand-new customers
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "
    return df
