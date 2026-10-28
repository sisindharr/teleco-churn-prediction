# 18 interview questions and answers (about this project)

**1. Walk me through the project end to end.**
I framed churn as a binary classification problem. I cleaned the Kaggle Telco data (fixed `TotalCharges`, merged redundant categories, encoded the target), explored churn drivers, engineered features, and built one scikit-learn pipeline (scaling + one-hot encoding + model). I compared Logistic Regression, Random Forest and XGBoost with 5-fold cross-validation, picked the winner on cross-validated ROC-AUC, and reported final metrics on a held-out test set. Then I wrapped the model in a Streamlit app with risk levels, recommendations and a what-if tool, and deployed it with CI.

**2. Why not just use accuracy?**
About 27% of customers churn, so a model that predicts "no churn" for everyone is about 73% accurate and useless. I look at recall (how many churners we catch), precision (how many alerts are right), F1 (their balance) and ROC-AUC (ranking quality independent of threshold).

**3. How did you handle class imbalance?**
I used class weights: `class_weight="balanced"` for Logistic Regression, `"balanced_subsample"` for Random Forest and `scale_pos_weight` for XGBoost, plus stratified splits and folds so the churn share is stable. Alternatives are SMOTE or threshold tuning. I preferred weights because they do not create synthetic customers and fit inside the pipeline without leakage risk.

**4. How did you prevent data leakage?**
All preprocessing lives in a `Pipeline` fitted only on training folds. The test set is split off before any fitting. Model selection uses cross-validation on the training data, not the test set, so the test score is an unbiased estimate. I also avoided features built from the target.

**5. What was wrong with `TotalCharges`, and how did you fix it?**
It was stored as text because 11 rows contained a blank string. All of them had tenure 0, meaning customers who had not been billed yet, so I set their total to 0 rather than dropping them or using the mean. Domain reasoning beats a blind imputation.

**6. Explain precision vs recall in business terms. How would you choose the threshold?**
Recall is the share of real churners we reach; a miss costs months of revenue. Precision is the share of offers that go to real churners; a miss costs a discount or call. If a retention offer is cheap relative to customer lifetime value, favour recall and use a lower threshold. In practice I would estimate cost per false positive and per false negative and pick the threshold that maximises expected profit, using out-of-fold predictions.

**7. Why one-hot encode and scale? Do tree models need it?**
Models need numbers; one-hot encoding avoids implying a fake order among categories. Scaling matters for Logistic Regression (regularisation and coefficient comparability). Trees do not need scaling, but I use a single shared preprocessor for consistency, and scaling does not hurt them.

**8. Compare Logistic Regression, Random Forest and XGBoost.**
Logistic Regression is linear, fast and interpretable through coefficients, a strong baseline on tabular data like this. Random Forest averages many decorrelated trees (bagging), reducing variance and capturing interactions. XGBoost builds trees sequentially to correct earlier errors (boosting) with regularisation, and often wins on tabular data, but needs more tuning. On this dataset the three can score close together, which is a useful finding: a simpler model may be preferable if it is nearly as good.

**9. Which features did you engineer and why?**
Tenure groups (new customers churn most), number of add-on services (stickiness), price per service, auto-pay flag (less friction), and a family flag. Each came from a business hypothesis I checked in EDA. I would measure their value with an ablation (train with and without them).

**10. How do you interpret feature importance, and what are its limits?**
The project shows model feature importance and SHAP. SHAP estimates how an input pushes an individual prediction up or down; the global summary averages absolute contributions across a customer sample. Correlated features can share credit, and neither SHAP nor feature importance proves causation. Here SHAP explains the uncalibrated base model, while the app uses the calibrated model for probabilities.

**11. What is cross-validation and why stratified?**
It trains and validates on several different splits (5 folds here) to give a more stable estimate and a standard deviation. Stratified folds keep the churn proportion equal in every fold, which matters with imbalanced classes.

**12. How would you tune hyper-parameters?**
The training script has a `--tune` flag that uses `RandomizedSearchCV` over sensible ranges (depth, learning rate, number of trees, regularisation, `C`). Random search covers the space more efficiently than a full grid. I would tune inside cross-validation on the training set only, and consider Bayesian optimisation (Optuna) for larger budgets.

**13. How are probabilities affected by class weights, and how do you handle it?**
Class weighting can distort raw probability estimates. This project reserves a calibration split from the training data, compares sigmoid and isotonic calibration using Brier score, and selects the lower-scoring method. A reliability curve is evaluated on the untouched test set. Calibration improves probability interpretation but should be monitored as customer behavior changes.

**14. What could break in production, and how would you monitor it?**
Risks: library version mismatch with the pickled model (the app warns about this), schema changes, unseen categories (handled with `handle_unknown="ignore"`), and **data drift** as pricing or products change. I would log inputs and predictions, track drift (for example Population Stability Index), compare predicted risk with actual churn once outcomes arrive, and schedule retraining when performance drops.

**15. How would you prove the business value, and what would you do next?**
Offline metrics are not business impact. I would run an A/B test: offer retention deals to a random half of prioritized customers and compare churn and revenue. Because the model predicts risk, not who will respond to an offer, the next step is **uplift modelling**. I would also monitor calibration and drift, validate the CLV assumptions, and add fairness checks. SHAP and the "what-if" tool describe learned associations, not causal proof.

**16. Why calibrate churn probabilities, and how did you avoid leakage?**
Class weights help detect churners but can distort the probability scale. I held out a calibration slice from the training split, fitted sigmoid and isotonic calibrators against it, and selected the lower Brier score. The final test split was not used for fitting or selection; I used it only to report Brier scores and reliability curves. In this Kaggle run isotonic was selected, and test Brier score improved from 0.166 to 0.139.

**17. What does SHAP explain in this app?**
SHAP estimates how each model input contributes to an individual score and summarizes average impact across a sample of customers. I run it on the uncalibrated base pipeline because that is the model whose feature-level behavior is directly exposed. The app still uses the calibrated pipeline for customer probabilities. SHAP describes associations learned by the model; it does not prove that changing a feature will change churn.

**18. Why rank customers by revenue at risk instead of churn probability alone?**
Probability-only ranking gives highest priority to the most likely churners, regardless of customer value. This app estimates CLV as monthly charges times gross margin times expected remaining months, then computes revenue at risk as calibrated churn probability times CLV. That can prioritize a valuable customer with moderate risk over a low-value customer with higher risk. The margin and remaining-month assumptions are adjustable estimates, so teams should validate them before allocating campaign budget.
