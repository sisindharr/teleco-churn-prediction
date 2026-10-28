"""Step 2b - Exploratory Data Analysis (EDA) with Seaborn & Matplotlib.

    python -m src.eda

EDA = looking at the data before modelling to understand it. Figures are saved
to reports/figures/ and the main findings are printed to the console.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import seaborn as sns

from .config import FIGURES_DIR, RAW_DATA
from .data_cleaning import clean_data, load_raw

PALETTE = {0: "#6366F1", 1: "#F43F5E"}


def _save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=150)
    plt.close(fig)


def churn_rate_table(df, col):
    return (df.groupby(col)["Churn"].agg(churn_rate="mean", customers="size")
              .sort_values("churn_rate", ascending=False).round(3))


def run(data_path=RAW_DATA) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    df = clean_data(load_raw(data_path))

    print(f"Rows: {len(df):,} | Columns: {df.shape[1]} | Overall churn rate: {df['Churn'].mean():.1%}")
    print(f"Missing values after cleaning: {int(df.isna().sum().sum())}")
    print("\nNumeric summary:\n", df[["tenure", "MonthlyCharges", "TotalCharges"]].describe().round(1))

    # 1) Class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x="Churn", hue="Churn", palette=PALETTE, legend=False, ax=ax)
    ax.set_xticks([0, 1], ["Retained", "Churned"])
    ax.set_title("How many customers churn?")
    _save(fig, "01_churn_distribution")

    # 2) Churn rate by the strongest categorical drivers
    for i, col in enumerate(["Contract", "InternetService", "PaymentMethod", "TechSupport"], start=2):
        table = churn_rate_table(df, col)
        print(f"\nChurn rate by {col}:\n{table}")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=table.index, y=table["churn_rate"] * 100, color="#6366F1", ax=ax)
        ax.set_ylabel("Churn rate (%)")
        ax.set_xlabel("")
        ax.set_title(f"Churn rate by {col}")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        _save(fig, f"0{i}_churn_by_{col.lower()}")

    # 3) Numeric distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
        sns.histplot(data=df, x=col, hue="Churn", palette=PALETTE, kde=True, element="step", ax=ax)
        ax.set_title(f"{col} by churn")
    _save(fig, "06_numeric_distributions")

    # 4) Box plot: monthly charges
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", hue="Churn", palette=PALETTE, legend=False, ax=ax)
    ax.set_xticks([0, 1], ["Retained", "Churned"])
    ax.set_title("Monthly charges vs churn")
    _save(fig, "07_monthly_charges_boxplot")

    # 5) Correlation heatmap (numeric columns only)
    corr = df[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn"]].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Correlation heatmap")
    _save(fig, "08_correlation_heatmap")

    print(f"\nFigures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    run()
