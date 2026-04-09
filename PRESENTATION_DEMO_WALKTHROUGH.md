# Presentation Demo Walkthrough (2-3 Minutes)

## Goal
Show an end-to-end, evidence-backed demonstration of the pipeline outputs in the notebook.

## Demo Order
1. Open `notebooks/fraud_detection_pipeline.ipynb`.
2. Show class imbalance output and EDA plots.
3. Show baseline results table.
4. Show threshold-optimization plot and confusion matrix.
5. Show robustness tables (CV summary + seed sensitivity).
6. End on final conclusion section.

## Script (Suggested Timing)

### 0:00 - 0:25 | Problem and Dataset
- "This project targets credit card fraud detection where fraud is only 0.1727% of transactions."
- "Because of this imbalance, I evaluate precision, recall, F1, ROC-AUC, and PR-AUC instead of accuracy."

### 0:25 - 0:55 | EDA Snapshot
- "Here is the class distribution showing extreme imbalance."
- "Amount is right-skewed, and the strongest correlations with fraud include V17, V14, and V12."

### 0:55 - 1:25 | Model Comparison
- "I trained four classical models under a consistent leak-safe pipeline with SMOTE."
- "Random Forest produced the strongest baseline F1 of 0.8367 and PR-AUC of 0.8766."

### 1:25 - 2:00 | Threshold Optimization
- "Instead of default threshold 0.5, I selected threshold 0.48 by policy: maximize recall while precision stays above 0.80."
- "At threshold 0.48, precision is 0.8333, recall is 0.8673, F1 is 0.85."
- "Confusion matrix shows 85 true frauds detected with 13 misses."

### 2:00 - 2:30 | Robustness and Final Takeaway
- "Cross-validation is stable, but a train-test gap indicates overfitting risk, so recalibration and monitoring are necessary."
- "Overall, this is a practical decision-support model with strong imbalanced-learning performance."

## Backup Talking Points (if asked)
- Why Random Forest: best precision-recall balance at baseline.
- Why tuning did not increase test F1: candidate grid did not improve generalization in this split.
- Why thresholding matters: operational tradeoff control is more important than fixed 0.5 in fraud tasks.
- Why not deep learning: project scope emphasizes classical ML aligned with course modules.
