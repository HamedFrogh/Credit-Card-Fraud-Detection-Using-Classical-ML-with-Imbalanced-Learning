# CSCI 597 Final Project Proposal

**Student:** Hamed  
**Project Title:** Credit Card Fraud Detection Using Classical ML with Imbalanced Learning

## 1. Problem Statement
Credit card fraud detection is a high-impact, real-world binary classification problem with extreme class imbalance (approximately **0.17%** fraudulent transactions). In this setting, standard accuracy can be misleading because a model that predicts all transactions as non-fraud can still achieve very high accuracy. This project focuses on building and evaluating robust classical machine learning pipelines that prioritize minority-class detection performance while controlling false positives.

## 2. Dataset
**Kaggle Credit Card Fraud Detection Dataset**
- **Size:** 284,807 transactions
- **Features:** 31 columns (including PCA-transformed anonymized features, `Time`, `Amount`, and `Class`)
- **Target:** `Class` (0 = non-fraud, 1 = fraud)
- **Source:** Publicly available on Kaggle

## 3. Objectives
1. Develop a reproducible end-to-end fraud detection pipeline for highly imbalanced data.
2. Compare multiple classical ML models under consistent preprocessing and evaluation settings.
3. Quantify trade-offs between precision and recall for fraud detection.
4. Tune the best-performing model and interpret key predictive features.

## 4. Methodology
### 4.1 Data Exploration and Preprocessing
- Perform exploratory data analysis (class distribution, feature summary, correlation inspection).
- Apply preprocessing with `StandardScaler` where appropriate.
- Use a **stratified train/test split** to preserve class proportions.

### 4.2 Imbalance Handling
- Apply **SMOTE** on the training data only to avoid data leakage.
- Compare baseline and imbalance-handled performance where relevant.

### 4.3 Model Training
Train and compare the following models:
- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

### 4.4 Evaluation Strategy
Use metrics appropriate for imbalanced classification:
- Precision
- Recall
- F1-score
- AUC-ROC
- Precision-Recall curve (and PR-AUC)

Primary model selection will emphasize fraud-detection usefulness (high recall with acceptable precision), not raw accuracy.

### 4.5 Hyperparameter Tuning
- Run `GridSearchCV` on the best-performing model from the comparison stage.
- Use cross-validation with a scoring metric aligned to class imbalance (e.g., F1, recall, or PR-AUC).

### 4.6 Model Interpretation
- Conduct feature importance analysis (tree-based importance and/or coefficient analysis for linear models).
- Discuss which features contribute most to fraud prediction and any practical implications.

## 5. Course Alignment
This project directly aligns with CSCI 597 topics:
- **Linear models (Weeks 2-3):** Logistic Regression
- **Model evaluation and bias-variance (Week 4):** metric selection, cross-validation, tuning
- **ML pipeline design (Week 4):** preprocessing, splitting, leakage prevention
- **Ensemble methods (Weeks 5-6):** Random Forest, Gradient Boosting
- **SVM (Week 7):** Support Vector Machine modeling

It also supports **CLOs 1-6** through end-to-end implementation, evaluation, and interpretation of machine learning methods.

## 6. Expected Deliverables
1. **Jupyter Notebook** containing a fully reproducible pipeline (EDA, preprocessing, modeling, evaluation, and tuning).
2. **Project Report** summarizing methodology, results, analysis, and conclusions.
3. **Presentation** highlighting motivation, approach, key findings, and lessons learned.

## 7. Success Criteria
The project will be considered successful if it:
- Demonstrates a complete and reproducible classical ML workflow.
- Shows clear comparative analysis across all four models.
- Uses imbalance-aware metrics to justify model choice.
- Provides interpretable insights into model behavior and feature impact.
