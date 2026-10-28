"""SHAP explanations for the uncalibrated base model."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from .config import ALL_FEATURES, BASE_MODEL_PATH, PROCESSED_DATA
from .features import add_features
from .predict import normalize_records
from .train import pretty_feature_name


@st.cache_resource(show_spinner=False)
def _explanation_context(base_model_mtime: int, data_mtime: int):
    import shap

    base_model = joblib.load(BASE_MODEL_PATH)
    data = pd.read_csv(PROCESSED_DATA)
    sample = data.sample(n=min(100, len(data)), random_state=42)
    sample_features = sample[ALL_FEATURES]
    preprocessor = base_model.named_steps["preprocess"]
    background = preprocessor.transform(sample_features)
    estimator = base_model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        explainer = shap.TreeExplainer(estimator, data=background, feature_perturbation="interventional")
    else:
        explainer = shap.LinearExplainer(estimator, background)
    names = [pretty_feature_name(name) for name in preprocessor.get_feature_names_out()]
    return preprocessor, explainer, names


def _cached_explanation_context():
    return _explanation_context(
        BASE_MODEL_PATH.stat().st_mtime_ns,
        PROCESSED_DATA.stat().st_mtime_ns,
    )


def _positive_class_values(values, n_rows: int, n_features: int) -> np.ndarray:
    """Normalize SHAP's list and ndarray binary-class output conventions."""
    if isinstance(values, list):
        if len(values) > 1:
            values = values[1]
        else:
            values = values[0]
    elif hasattr(values, "values"):
        values = values.values

    values = np.asarray(values)
    if values.ndim == 3:
        if values.shape[:2] == (n_rows, n_features):
            values = values[:, :, 1] if values.shape[2] > 1 else values[:, :, 0]
        elif values.shape[1:] == (n_rows, n_features):
            values = values[1] if values.shape[0] > 1 else values[0]
    if values.shape != (n_rows, n_features):
        raise ValueError(f"Unexpected SHAP output shape: {values.shape}")
    return values


def _shap_values(features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    preprocessor, explainer, names = _cached_explanation_context()
    matrix = preprocessor.transform(features[ALL_FEATURES])
    values = _positive_class_values(explainer.shap_values(matrix), len(features), len(names))
    return values, names


def explain_record(record: dict, top_n: int = 8) -> pd.DataFrame:
    """Return local signed contributions for one customer's positive-class score."""
    features = add_features(normalize_records(pd.DataFrame([record])))
    values, names = _shap_values(features)
    result = pd.DataFrame({"feature": names, "impact": values[0]})
    result["effect"] = np.where(result["impact"] >= 0, "Raises churn score", "Lowers churn score")
    return (result.reindex(result["impact"].abs().sort_values(ascending=False).index)
            .head(top_n).reset_index(drop=True))


def global_importance(data: pd.DataFrame, sample_size: int = 500) -> pd.DataFrame:
    """Return mean absolute SHAP impact for a bounded, reproducible data sample."""
    sample = data.sample(n=min(sample_size, len(data)), random_state=42)
    values, names = _shap_values(sample)
    result = pd.DataFrame({"feature": names, "importance": np.abs(values).mean(axis=0)})
    return result.sort_values("importance", ascending=False).reset_index(drop=True)
