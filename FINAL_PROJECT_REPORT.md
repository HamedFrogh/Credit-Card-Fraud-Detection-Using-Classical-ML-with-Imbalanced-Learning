# Final Project Report

## CSCI 597: Credit Card Fraud Detection Using Classical ML with Imbalanced Learning
**Student:** Hamed  
**Date:** April 8, 2026

## Abstract
Credit card fraud detection is a highly imbalanced classification problem where standard accuracy can be misleading. This project implements a complete classical machine learning pipeline on the Kaggle Credit Card Fraud dataset (284,807 records, 31 features, 0.1727% fraud rate). The workflow includes preprocessing, SMOTE-based imbalance handling, model comparison (Logistic Regression, Random Forest, Gradient Boosting, SVM), threshold optimization, hyperparameter tuning, and robustness checks. Random Forest achieved the strongest baseline F1 performance (0.8367), while threshold tuning provided an operationally improved balance (Precision 0.8333, Recall 0.8673, F1 0.8500 at threshold 0.48). Results show strong predictive performance with practical caveats regarding PCA feature interpretability, class imbalance sensitivity, and model monitoring needs.

## 1. Problem Statement and Motivation
Fraud detection requires identifying rare positive events with high cost of missed detection. In the target dataset, fraudulent transactions represent only 492 out of 284,807 samples (0.1727%). In this setting, a naive classifier can achieve very high accuracy while failing to detect fraud. Therefore, the project focuses on precision, recall, F1, ROC-AUC, and PR-AUC rather than accuracy.

## 2. Dataset
- Source: Kaggle Credit Card Fraud Detection Dataset
- Total rows: 284,807
- Fraud rows: 492
- Features: 31 columns including `Time`, `Amount`, PCA-transformed anonymized variables (`V1` to `V28`), and target `Class`
- Target: `Class` (0 = non-fraud, 1 = fraud)

## 3. Methodology
## 3.1 Pipeline and Preprocessing
- Stratified train/test split to preserve class ratio.
- Standardization applied where appropriate (Logistic Regression, SVM).
- SMOTE applied only in training pipeline stages to avoid leakage.

## 3.2 Models Evaluated
- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (RBF kernel)

## 3.3 Evaluation Metrics
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC (Average Precision)

## 3.4 Tuning and Post-Modeling
- `GridSearchCV` on the best baseline model.
- Threshold sweep from 0.05 to 0.95 to select policy-based operating point.
- Robustness checks with cross-validation, train-vs-test comparison, and alternate split seeds.

## 4. Exploratory Data Analysis Summary
- Class imbalance is severe (0.1727% fraud).
- `Amount` is heavily right-skewed with large outliers.
- Top absolute correlations with fraud class include `V17`, `V14`, `V12`, `V10`, `V16`, `V3`, `V7`, and `V11`.

Key EDA figures are available in notebook outputs and plotted in `notebooks/fraud_detection_pipeline.ipynb`.

## 5. Baseline Model Results
Strongest baseline model by F1 was Random Forest.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.8367 | 0.8367 | 0.8367 | 0.9725 | 0.8766 |
| Gradient Boosting | 0.1845 | 0.8980 | 0.3061 | 0.9798 | 0.7563 |
| SVM | 0.0854 | 0.8673 | 0.1555 | 0.9647 | 0.5955 |
| Logistic Regression | 0.0578 | 0.9184 | 0.1088 | 0.9708 | 0.7245 |

Source table: `outputs/baseline_results.csv`.

## 6. Hyperparameter Tuning
Best model selected for tuning: Random Forest.

- Best parameters: `{'model__max_depth': None, 'model__min_samples_split': 2, 'model__n_estimators': 200}`
- CV best F1: 0.8511
- Test metrics after tuning:
  - Precision: 0.8265
  - Recall: 0.8265
  - F1: 0.8265
  - ROC-AUC: 0.9685
  - PR-AUC: 0.8759

Observation: tuned test F1 was slightly below baseline F1, indicating limited additional generalization benefit under the tested grid.

Source: `outputs/tuned_best_model_summary.csv`.

## 7. Threshold Optimization
A policy-based threshold was selected for operational usage:
- Policy: maximize recall with precision >= 0.80
- Selected threshold: 0.48
- Metrics at selected threshold:
  - Precision: 0.8333
  - Recall: 0.8673
  - F1: 0.8500
- Confusion matrix at threshold 0.48:
  - TN = 56846
  - FP = 18
  - FN = 13
  - TP = 85

This threshold improved the recall-oriented operating balance compared with default threshold 0.50 while keeping precision above policy target.

Sources:
- `outputs/threshold_optimization_random_forest.csv`
- Threshold/CM plot from `notebooks/fraud_detection_pipeline.ipynb`

## 8. Robustness and Generalization Checks
## 8.1 Cross-Validation Stability (3-Fold, Random Forest)
- Precision mean/std: 0.8793 / 0.0110
- Recall mean/std: 0.8096 / 0.0312
- F1 mean/std: 0.8425 / 0.0120
- ROC-AUC mean/std: 0.9758 / 0.0085
- PR-AUC mean/std: 0.8414 / 0.0131

## 8.2 Overfitting Indicator
- Train F1: 1.0000
- Test F1: 0.8265

The train-test gap indicates overfitting risk and supports continued threshold calibration and monitoring.

## 8.3 Seed Sensitivity
Test F1 under alternate split seeds:
- Seed 42: 0.8265
- Seed 7: 0.8632
- Seed 2026: 0.8691

Performance is reasonably strong across seeds, though split variability exists.

Sources:
- `outputs/robustness_cv_summary_random_forest.csv`
- `outputs/robustness_train_test_overfit_random_forest.csv`
- `outputs/robustness_seed_sensitivity_random_forest.csv`

## 9. Limitations and Future Work
- PCA-transformed feature space limits business-level interpretability of individual variables.
- Correlation and feature importance are associative, not causal.
- Class imbalance can amplify threshold sensitivity and false-positive cost tradeoffs.
- SMOTE introduces synthetic minority samples and may not represent all real fraud patterns.
- Temporal drift in fraud behavior requires periodic retraining and threshold recalibration.

Future work:
- Add time-aware validation and drift monitoring.
- Calibrate probabilities (Platt/Isotonic) before threshold selection.
- Evaluate cost-sensitive learning with explicit fraud investigation cost matrix.
- Extend to API + dashboard deployment for interactive scoring.

## 10. Conclusion
This project demonstrates a robust classical ML pipeline for extreme class imbalance in fraud detection. Random Forest provided the strongest baseline F1 and maintained strong ROC-AUC/PR-AUC performance. Threshold optimization produced a more practical operating point (t = 0.48) with high recall and controlled precision. Robustness analyses confirmed generally stable performance while identifying expected overfitting and split sensitivity considerations. Overall, the system is suitable as a decision-support fraud screening model with periodic retraining and monitoring.

## References
1. Dal Pozzolo, A., et al. "Calibrating Probability with Undersampling for Unbalanced Classification." IEEE Symposium Series on Computational Intelligence, 2015.
2. Kaggle. "Credit Card Fraud Detection Dataset." https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
3. Pedregosa, F., et al. "Scikit-learn: Machine Learning in Python." Journal of Machine Learning Research, 2011.
4. Chawla, N. V., et al. "SMOTE: Synthetic Minority Over-sampling Technique." Journal of Artificial Intelligence Research, 2002.
