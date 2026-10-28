"""Step 3 - Model training, comparison and evaluation.

Run from the project root:

    python -m src.train            # fast: sensible default settings
    python -m src.train --tune     # slower: random hyper-parameter search

What this script does (in plain English)
----------------------------------------
1. Loads and cleans the data, then adds engineered features.
2. Splits it: 80% to learn from (train), 20% kept hidden to grade the models (test).
3. Builds one preprocessing recipe (scale numbers, one-hot encode categories).
4. Trains three models: Logistic Regression, Random Forest, XGBoost.
5. Uses 5-fold cross-validation on the training data to pick the best model fairly.
6. Reports Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, ROC curve
   and feature importance on the untouched test set.
7. Saves the winning model + a metrics.json file that the Streamlit app reads.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .config import (BASE_MODEL_PATH, CATEGORICAL_FEATURES, CV_FOLDS, DECISION_THRESHOLD,
                     METRICS_PATH, MODEL_PATH, NUMERIC_FEATURES, PROCESSED_DATA,
                     RANDOM_STATE, RAW_DATA, TEST_SIZE)
from .data_cleaning import clean_data, load_raw
from .features import add_features, split_xy

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train")


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """Scale numeric columns and one-hot encode categorical columns.

    * StandardScaler: puts numbers on a similar scale (important for Logistic Regression).
    * OneHotEncoder : turns text categories into 0/1 columns the model can read.
      handle_unknown="ignore" means a brand-new category will not crash the app.
    """
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


def get_models(scale_pos_weight: float) -> dict:
    """Three models, all configured to handle the class imbalance (~27% churners)."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, C=0.5, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE),
    }


PARAM_DISTRIBUTIONS = {
    "Logistic Regression": {"model__C": [0.01, 0.1, 0.5, 1, 5, 10]},
    "Random Forest": {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 8, 12],
        "model__min_samples_leaf": [2, 5, 10],
    },
    "XGBoost": {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [2, 3, 4, 5],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
    },
}


def pretty_feature_name(raw: str) -> str:
    """'cat__Contract_Month-to-month' -> 'Contract = Month-to-month'."""
    body = raw.split("__", 1)[1]
    if raw.startswith("num__"):
        return body
    for col in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
        if body.startswith(col + "_"):
            return f"{col} = {body[len(col) + 1:]}"
    return body


def feature_importance(pipe: Pipeline, top_n: int = 25) -> list[dict]:
    """Which inputs matter most? Trees give importances, Logistic Regression gives coefficients."""
    names = [pretty_feature_name(n) for n in pipe.named_steps["preprocess"].get_feature_names_out()]
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values, signed = np.asarray(model.feature_importances_, dtype=float), None
    else:
        signed = model.coef_[0]
        values = np.abs(signed)
    frame = pd.DataFrame({"feature": names, "importance": values})
    if signed is not None:
        frame["coefficient"] = signed
    frame = frame.sort_values("importance", ascending=False).head(top_n)
    return json.loads(frame.round(5).to_json(orient="records"))


def evaluate(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Score a fitted pipeline on the hidden test set."""
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= DECISION_THRESHOLD).astype(int)
    fpr, tpr, _ = roc_curve(y_test, proba)
    return {
        "metrics": {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba),
        },
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),  # [[TN, FP], [FN, TP]]
        "roc_curve": {"fpr": np.round(fpr, 4).tolist(), "tpr": np.round(tpr, 4).tolist()},
    }


def _calibrator(base_model: Pipeline, method: str):
    """Build a calibrator around an already-fitted model without refitting it."""
    version_parts = tuple(int(part) for part in sklearn.__version__.split(".")[:2])
    if version_parts >= (1, 6):
        from sklearn.frozen import FrozenEstimator

        estimator = FrozenEstimator(base_model)
        return CalibratedClassifierCV(estimator=estimator, method=method)
    return CalibratedClassifierCV(estimator=base_model, method=method, cv="prefit")


def _reliability_points(y_true: pd.Series, probabilities: np.ndarray) -> dict:
    observed, predicted = calibration_curve(
        y_true, probabilities, n_bins=10, strategy="uniform")
    return {
        "mean_predicted_probability": np.round(predicted, 4).tolist(),
        "fraction_of_positives": np.round(observed, 4).tolist(),
    }


# ----------------------------------------------------------------------------
# Main workflow
# ----------------------------------------------------------------------------
def run(data_path=RAW_DATA, tune: bool = False) -> dict:
    # 1) Data
    df = add_features(clean_data(load_raw(data_path)))
    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA, index=False)
    logger.info("Processed data saved to %s (%d rows)", PROCESSED_DATA, len(df))

    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    X_fit, X_calibration, y_fit, y_calibration = train_test_split(
        X_train, y_train, test_size=0.20, stratify=y_train, random_state=RANDOM_STATE)
    logger.info("Train: %d rows | Calibration: %d rows | Test: %d rows | Churn rate: %.1f%%",
                len(X_fit), len(X_calibration), len(X_test), 100 * y.mean())

    scale_pos_weight = float((y_fit == 0).sum() / (y_fit == 1).sum())
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results: dict = {}
    fitted: dict[str, Pipeline] = {}

    # 2) Train + compare
    for name, model in get_models(scale_pos_weight).items():
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", model)])

        if tune:
            logger.info("Tuning %s ...", name)
            search = RandomizedSearchCV(
                pipe, PARAM_DISTRIBUTIONS[name], n_iter=10, scoring="roc_auc",
                cv=3, random_state=RANDOM_STATE, n_jobs=-1)
            search.fit(X_fit, y_fit)
            pipe = search.best_estimator_
            logger.info("  best params: %s", search.best_params_)

        cv_scores = cross_val_score(pipe, X_fit, y_fit, cv=cv, scoring="roc_auc", n_jobs=1)
        pipe.fit(X_fit, y_fit)

        results[name] = {
            **evaluate(pipe, X_test, y_test),
            "cv_roc_auc_mean": float(cv_scores.mean()),
            "cv_roc_auc_std": float(cv_scores.std()),
            "feature_importance": feature_importance(pipe),
        }
        fitted[name] = pipe
        m = results[name]["metrics"]
        logger.info("%-20s CV AUC %.3f | Test: acc %.3f prec %.3f rec %.3f f1 %.3f auc %.3f",
                    name, cv_scores.mean(), m["accuracy"], m["precision"], m["recall"], m["f1"], m["roc_auc"])

    # 3) Pick the winner using cross-validation (NOT the test set, to avoid peeking)
    best_name = max(results, key=lambda n: results[n]["cv_roc_auc_mean"])
    logger.info("Best model by CV ROC-AUC: %s", best_name)

    # 4) Calibrate on a held-out slice of training data, never on the test set.
    base_model = fitted[best_name]
    calibration_scores = {}
    calibrators = {}
    for method in ("sigmoid", "isotonic"):
        calibrated = _calibrator(base_model, method)
        calibrated.fit(X_calibration, y_calibration)
        calibrators[method] = calibrated
        calibration_scores[method] = float(
            brier_score_loss(y_calibration, calibrated.predict_proba(X_calibration)[:, 1]))
    selected_method = min(calibration_scores, key=calibration_scores.get)
    calibrated_model = calibrators[selected_method]
    logger.info("Calibration: %s (sigmoid Brier %.4f, isotonic Brier %.4f)",
                selected_method, calibration_scores["sigmoid"], calibration_scores["isotonic"])

    before_test_probabilities = base_model.predict_proba(X_test)[:, 1]
    after_test_probabilities = calibrated_model.predict_proba(X_test)[:, 1]
    calibration = {
        "method": selected_method,
        "selection_brier_scores": calibration_scores,
        "brier_before": float(brier_score_loss(y_test, before_test_probabilities)),
        "brier_after": float(brier_score_loss(y_test, after_test_probabilities)),
        "reliability_before": _reliability_points(y_test, before_test_probabilities),
        "reliability_after": _reliability_points(y_test, after_test_probabilities),
    }

    # 5) Save artefacts
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated_model, MODEL_PATH)
    joblib.dump(base_model, BASE_MODEL_PATH)
    summary = {
        "best_model": best_name,
        "threshold": DECISION_THRESHOLD,
        "calibration": calibration,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "rows": int(len(df)), "churn_rate": float(y.mean()),
            "train_size": int(len(X_train)), "test_size": int(len(X_test)),
            "n_features_raw": len(X.columns),
        },
        "versions": {
            "python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__, "pandas": pd.__version__,
        },
        "tuned": tune,
        "models": results,
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2))
    logger.info("Saved calibrated model -> %s, base model -> %s and metrics -> %s",
                MODEL_PATH, BASE_MODEL_PATH, METRICS_PATH)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare churn models.")
    parser.add_argument("--data", default=str(RAW_DATA), help="Path to the raw Telco CSV")
    parser.add_argument("--tune", action="store_true", help="Run randomized hyper-parameter search")
    args = parser.parse_args()
    run(args.data, args.tune)


if __name__ == "__main__":
    main()
