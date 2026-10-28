# Project guide: the business case and every step explained

## 1. Business problem

A telecom company sells phone and internet plans on monthly or yearly contracts. Each month some customers cancel. This is called **churn**.

- Replacing a lost customer (marketing, onboarding, discounts) usually costs much more than keeping one.
- The retention team has limited time and budget. They cannot offer discounts to everyone.

**Question:** which current customers are most likely to leave, and what could we offer them?

## 2. Objectives

1. Build a model that gives each customer a **churn probability**.
2. Find the **main drivers** of churn so the business understands the "why".
3. Turn scores into **actions**: risk levels, a call list, recommended offers.
4. Ship it as an app that non-technical colleagues can use.

## 3. Expected business impact

| Lever | How the project helps |
|---|---|
| Revenue retained | Saving even a modest share of high-risk customers protects months of recurring revenue. |
| Cost efficiency | Offers go to the customers most likely to leave, not to everybody. |
| Better products | Drivers (for example contract type or support) point to fixable problems. |

Impact depends on how many offers succeed. The Home page estimator lets you test assumptions (success rate, cost per offer, months of revenue kept). Treat the output as a scenario, not a forecast.

## 4. The data

The Kaggle *Telco Customer Churn* dataset (IBM sample): about 7,000 customers and 21 columns.

- **Who:** gender, senior citizen, partner, dependents
- **What they buy:** phone, multiple lines, internet type, security, backup, device protection, tech support, streaming TV/movies
- **How they pay:** contract, paperless billing, payment method, monthly and total charges
- **How long:** `tenure` (months)
- **Target:** `Churn` (Yes/No)

## 5. Step by step

### Step 1: Data cleaning (`src/data_cleaning.py`)
Raw data is messy, and models are only as good as their inputs.
- **TotalCharges is text.** A few rows hold a blank space. They are brand-new customers (tenure 0) who have not been billed, so the correct value is 0.
- **"No internet service" and "No phone service"** carry the same meaning as "No" for the related add-on. We merge them to avoid duplicate categories.
- **Churn "Yes"/"No" becomes 1/0** because models need numbers.
- We remove duplicates and strip stray whitespace.

### Step 2: Exploratory Data Analysis (`src/eda.py`, notebook)
EDA means asking the data questions with charts *before* modelling.
- How many customers churn? (about 1 in 4: this is **class imbalance**)
- Does churn change with contract, tenure, internet type, payment method, price?
- Which numeric columns move together?

Always read the numbers from your own run. Typical patterns in this dataset: month-to-month contracts, short tenure, fiber internet and electronic-check payments sit with higher churn.

### Step 3: Feature engineering (`src/features.py`)
Create columns that express business ideas.

| Feature | Idea |
|---|---|
| `tenure_group` | New customers behave differently from loyal ones |
| `num_addon_services` | More add-ons make a customer harder to lose |
| `monthly_per_service` | Price paid for what they actually use |
| `has_autopay` | Automatic payment removes a "should I cancel?" moment each month |
| `has_family` | Partner or dependents often mean a more stable household |

The **same function** runs in training and in the app, so predictions never drift because of different preprocessing.

### Step 4: Preprocessing (`src/train.py`)
- **Train/test split (80/20, stratified):** the test set is a final exam the models never study from.
- **StandardScaler** on numbers: puts them on a similar scale (matters for Logistic Regression).
- **OneHotEncoder** on text categories: turns "Contract = Two year" into a 0/1 column. `handle_unknown="ignore"` means a new category cannot crash the app.
- Everything lives in one scikit-learn **Pipeline**, fitted on training data only. This prevents **data leakage**.

### Step 5: Model training
| Model | Intuition | Why include it |
|---|---|---|
| Logistic Regression | Weighs each feature and adds them up | Simple, fast, explainable baseline |
| Random Forest | Hundreds of decision trees vote | Captures non-linear patterns, robust |
| XGBoost | Trees built one after another, each fixing earlier errors | Often the strongest on tabular data |

**Class imbalance** is handled with class weights (`class_weight`, `scale_pos_weight`), so the models do not simply predict "no churn" for everyone.

### Step 6: Evaluation
We use 5-fold **cross-validation** on the training set to pick the winner (ROC-AUC), then report final scores on the hidden test set.

| Metric | Plain meaning | Why it matters here |
|---|---|---|
| Accuracy | Share of right predictions | Misleading when most customers stay |
| Precision | When we flag churn, how often are we right? | Low precision wastes offers |
| Recall | Of real churners, how many did we catch? | Low recall means lost revenue |
| F1 | Balance of precision and recall | Single summary number |
| ROC-AUC | How well risky customers rank above safe ones | Threshold-independent quality |
| Confusion matrix | Counts of right and wrong calls | Shows the types of mistakes |

### Step 6b: Probability calibration
Class weights help the model notice churners, but can make its raw probability numbers too high. Calibration adjusts the score scale so that, for example, customers assigned a probability near 30% churn at about that rate over time.

The training script holds a separate slice out of the training data for calibration. It compares sigmoid (Platt) and isotonic calibration using the **Brier score**; lower is better. The untouched test set is not used to fit or select the calibrator. A reliability curve compares each probability band with its observed churn frequency. The app uses the calibrated pipeline for predictions, what-if scenarios, and batch scoring.

In the current Kaggle run, isotonic calibration was selected; the held-out test Brier score changed from 0.166 before calibration to 0.139 after it. This is evidence for this test split, not a guarantee of future performance.

### Step 7: Feature importance and SHAP
The feature-importance list shows what the model generally uses. SHAP goes further: it estimates how each input pushes one customer's score higher or lower, and its global summary averages the size of those contributions across customers.

SHAP is calculated on the **uncalibrated base model** because its feature representation is directly explainable. The app still uses the calibrated model for probabilities. SHAP contributions describe model behavior and associations, not causes; correlated inputs can share or distort credit.

### Step 8: Risk levels and recommendations (`src/predict.py`)
- Low: below 30%. Medium: 30% to 60%. High: 60% or more.
- Rule-based recommendations turn a score into actions (for example, offer a longer contract to a month-to-month customer).
- The "what-if" tool re-scores the customer after a hypothetical change (such as a 1-year contract).

### Step 9: Customer lifetime value and priority (`src/value.py`)
Lifetime value (CLV) is estimated as:

`monthly charges × gross margin × expected remaining months`

Revenue at risk is the customer's **calibrated churn probability × estimated CLV**. The Home and Batch Scoring pages default to a 30% gross margin and 24 remaining months; both assumptions can be changed. Ranking by revenue at risk can put a lower-probability, higher-value customer ahead of a high-risk, low-value one. The estimates are planning aids, not guaranteed retained revenue or accounting values.

### Step 10: Batch scoring
The Batch Scoring page provides a raw Telco CSV template, checks required fields and numeric values, and applies the same cleaning rules as training. It scores and ranks rows in memory; uploaded data is not saved to disk or a database.

### Step 11: Deployment
The fitted pipeline is saved with `joblib`. The Streamlit app loads it once (cached), builds the input form, and calls `predict_proba`. See `DEPLOYMENT.md`.

## 6. Glossary

- **Churn**: a customer cancels.
- **Feature**: an input column for the model.
- **Pipeline**: a chain of preprocessing and model steps run in one go.
- **Cross-validation**: train and test several times on different splits for a steadier estimate.
- **Data leakage**: information from the test data sneaking into training, which inflates results.
- **Threshold**: probability above which we label a customer "will churn" (default 50%).
