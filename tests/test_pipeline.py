"""Automated tests. Run with:  pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src.config import ALL_FEATURES, CATEGORY_OPTIONS
from src.batch import score_batch
from src.data_cleaning import clean_data
from src.features import add_features
from src.predict import normalize_records, recommendations, risk_band
from src.value import add_customer_value
from tests.synthetic import make_synthetic_telco


@pytest.fixture(scope="module")
def raw():
    return make_synthetic_telco(800)


def test_cleaning_fixes_total_charges_and_target(raw):
    clean = clean_data(raw)
    assert clean["TotalCharges"].dtype.kind == "f"
    assert clean["TotalCharges"].isna().sum() == 0
    assert set(clean["Churn"].unique()) <= {0, 1}
    assert "No internet service" not in clean["TechSupport"].unique()
    assert "No phone service" not in clean["MultipleLines"].unique()


def test_feature_engineering_columns(raw):
    feats = add_features(clean_data(raw))
    assert set(ALL_FEATURES).issubset(feats.columns)
    assert feats["num_addon_services"].between(0, 6).all()
    assert feats["tenure_group"].isin(["0-12 months", "13-24 months", "25-48 months", "49+ months"]).all()


def test_risk_bands():
    assert risk_band(0.05) == "Low"
    assert risk_band(0.45) == "Medium"
    assert risk_band(0.85) == "High"


def test_shap_binary_output_uses_positive_class():
    from src.explain import _positive_class_values

    values = np.arange(12).reshape(2, 3, 2)
    result = _positive_class_values(values, n_rows=2, n_features=3)
    np.testing.assert_array_equal(result, values[:, :, 1])


def _sample_record():
    rec = {k: v[0] for k, v in CATEGORY_OPTIONS.items()}
    rec.update(tenure=3, MonthlyCharges=85.0, SeniorCitizen=0,
               Contract="Month-to-month", PaymentMethod="Electronic check")
    return rec


def test_normalize_records_enforces_business_rules():
    rec = {**_sample_record(), "InternetService": "No", "TechSupport": "Yes"}
    out = normalize_records(pd.DataFrame([rec]))
    assert out.loc[0, "TechSupport"] == "No"
    assert out.loc[0, "TotalCharges"] == pytest.approx(85.0 * 3)


def test_recommendations_for_risky_customer():
    recs = recommendations(_sample_record(), 0.8)
    titles = [r["title"] for r in recs]
    assert "Escalate to a retention specialist" in titles
    assert "Offer a longer contract" in titles


def test_customer_value_calculations_and_priority():
    customers = pd.DataFrame({
        "MonthlyCharges": [100.0, 50.0],
        "churn_prob": [0.2, 0.8],
    })
    valued = add_customer_value(customers)
    assert valued["clv"].tolist() == [720.0, 360.0]
    assert valued["revenue_at_risk"].tolist() == [144.0, 288.0]
    assert valued["priority_rank"].tolist() == [2, 1]


def test_customer_value_custom_assumptions_and_invalid_margin():
    customers = pd.DataFrame({"MonthlyCharges": [100.0], "churn_prob": [0.5]})
    valued = add_customer_value(customers, gross_margin=0.5, expected_remaining_months=12)
    assert valued.loc[0, "clv"] == 600.0
    assert valued.loc[0, "expected_remaining_months"] == 12
    with pytest.raises(ValueError, match="gross_margin"):
        add_customer_value(customers, gross_margin=1.5)


def test_batch_scoring_preserves_rows_and_ranks_by_revenue(raw):
    class FixedProbabilityModel:
        def predict_proba(self, features):
            churn = np.linspace(0.1, 0.9, len(features))
            return np.column_stack([1 - churn, churn])

    batch = raw.head(8).drop(columns="Churn")
    scored = score_batch(FixedProbabilityModel(), batch)
    assert len(scored) == len(batch)
    assert {"churn_probability", "risk_level", "clv", "revenue_at_risk", "priority_rank"} <= set(scored)
    assert scored["revenue_at_risk"].is_monotonic_decreasing
    assert scored["priority_rank"].tolist() == list(range(1, len(batch) + 1))


def test_training_end_to_end(tmp_path, monkeypatch, raw):
    """Train on synthetic data, then check the saved model scores a customer."""
    pytest.importorskip("xgboost")
    import joblib
    import json
    from src import train
    from src.predict import predict_records

    monkeypatch.setattr(train, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(train, "BASE_MODEL_PATH", tmp_path / "base_model.joblib")
    monkeypatch.setattr(train, "METRICS_PATH", tmp_path / "metrics.json")
    monkeypatch.setattr(train, "PROCESSED_DATA", tmp_path / "processed.csv")
    csv = tmp_path / "raw.csv"
    raw.to_csv(csv, index=False)

    summary = train.run(csv)
    assert set(summary["models"]) == {"Logistic Regression", "Random Forest", "XGBoost"}
    for res in summary["models"].values():
        assert 0.5 < res["metrics"]["roc_auc"] <= 1.0
        assert np.array(res["confusion_matrix"]).shape == (2, 2)
    assert json.loads((tmp_path / "metrics.json").read_text())["best_model"] == summary["best_model"]
    assert (tmp_path / "base_model.joblib").exists()
    assert summary["calibration"]["method"] in {"sigmoid", "isotonic"}
    assert 0.0 <= summary["calibration"]["brier_after"] <= 1.0

    model = joblib.load(tmp_path / "model.joblib")
    prob = predict_records(model, [_sample_record()])[0]
    assert 0.0 <= prob <= 1.0
