# Presentation Outline (10 Slides)

## Slide 1: Title and Context
- Title: Credit Card Fraud Detection Using Classical ML with Imbalanced Learning
- Student: Hamed
- Course: CSCI 597
- One-line goal: Build a robust fraud detection pipeline for extreme class imbalance.

Speaker note:
- Fraud rate is only 0.1727%, so metric choice and threshold design are critical.

## Slide 2: Problem Importance
- Credit card fraud is high cost and rare-event heavy.
- Accuracy is misleading in imbalanced settings.
- Objective: maximize useful fraud detection while controlling false positives.

Speaker note:
- A model can be >99% accurate and still detect almost no fraud.

## Slide 3: Dataset Overview
- Kaggle Credit Card Fraud dataset
- 284,807 transactions, 31 features, target `Class`
- Fraud cases: 492 (0.1727%)
- Features include PCA-transformed `V1` to `V28`, plus `Time` and `Amount`

Visual suggestion:
- Class imbalance bar chart from notebook EDA.

## Slide 4: Methodology Pipeline
- Stratified train/test split
- Scaling for LR and SVM
- SMOTE in training pipeline only (leak-safe)
- Models: Logistic Regression, Random Forest, Gradient Boosting, SVM
- Metrics: Precision, Recall, F1, ROC-AUC, PR-AUC

Visual suggestion:
- Simple pipeline diagram.

## Slide 5: EDA Highlights
- Severe class imbalance confirmed (0.1727% fraud)
- `Amount` is heavily right-skewed
- Top correlation features with fraud: V17, V14, V12, V10, V16, V3, V7, V11

Visual suggestion:
- Amount distribution + top-correlation bar chart.

## Slide 6: Baseline Model Comparison
| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.8367 | 0.8367 | 0.8367 | 0.9725 | 0.8766 |
| Gradient Boosting | 0.1845 | 0.8980 | 0.3061 | 0.9798 | 0.7563 |
| SVM | 0.0854 | 0.8673 | 0.1555 | 0.9647 | 0.5955 |
| Logistic Regression | 0.0578 | 0.9184 | 0.1088 | 0.9708 | 0.7245 |

Speaker note:
- Random Forest is strongest by balanced precision-recall tradeoff.

## Slide 7: Tuning and Threshold Optimization
- GridSearchCV best model: Random Forest
- Best params: `max_depth=None`, `min_samples_split=2`, `n_estimators=200`
- Tuned test F1: 0.8265
- Policy-based threshold chosen: maximize recall with precision >= 0.80
- Selected threshold: 0.48

Metrics at threshold 0.48:
- Precision: 0.8333
- Recall: 0.8673
- F1: 0.8500

Visual suggestion:
- Metric-vs-threshold plot + confusion matrix.

## Slide 8: Robustness and Reliability
- 3-fold CV (Random Forest):
  - F1 mean/std: 0.8425 / 0.0120
  - ROC-AUC mean/std: 0.9758 / 0.0085
- Overfitting check:
  - Train F1: 1.0000
  - Test F1: 0.8265
- Seed sensitivity F1 range: 0.8265 to 0.8691

Speaker note:
- Strong performance is stable, but train-test gap requires monitoring.

## Slide 9: Limitations and Future Work
- PCA features reduce direct business interpretability.
- Correlation and importance are not causal.
- Threshold is context-dependent and should be recalibrated.
- Future work:
  - Probability calibration
  - Time-aware validation and drift detection
  - Cost-sensitive optimization
  - API + frontend deployment

## Slide 10: Conclusion
- A classical ML pipeline can perform strongly for imbalanced fraud detection.
- Random Forest + SMOTE + threshold tuning provides practical decision support.
- Final takeaway: the model is effective when combined with continuous monitoring and periodic recalibration.

Closing line:
- "This project balances statistical rigor with operational usability for real fraud screening scenarios."
