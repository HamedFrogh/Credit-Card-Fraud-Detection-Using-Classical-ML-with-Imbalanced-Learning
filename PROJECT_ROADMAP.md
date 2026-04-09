# Project Roadmap

## CSCI 597 Final Project
**Student:** Hamed  
**Project:** Credit Card Fraud Detection Using Classical ML with Imbalanced Learning

## Phase 1: Setup and Data Readiness
1. Confirm Python environment and dependencies.
2. Place dataset at `data/creditcard.csv`.
3. Run baseline pipeline once end-to-end.

**Done when:**
- Command runs without errors:
  - `python -m src.train --data-path data/creditcard.csv --output-dir outputs`
- `outputs/baseline_results.csv` is generated.

## Phase 2: EDA and Data Understanding
1. Add class imbalance visualization.
2. Analyze `Amount` and `Time` distributions.
3. Add correlation analysis with fraud class.
4. Write short insights for each EDA plot.

**Done when:**
- Notebook contains EDA visuals and interpretation.
- Clear explanation of why accuracy is misleading.

## Phase 3: Preprocessing and Evaluation Protocol
1. Use stratified train/test split.
2. Apply scaling only for LR and SVM.
3. Apply SMOTE only on training folds to avoid leakage.
4. Add reproducibility details (seed, package versions).

**Done when:**
- Pipeline is leak-safe and reproducible.
- Baseline and SMOTE variants can be compared fairly.

## Phase 4: Baseline Model Benchmarking
1. Train these models:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - SVM
2. Evaluate with:
   - Precision
   - Recall
   - F1-score
   - ROC-AUC
   - PR-AUC
3. Save ROC and PR curves.

**Done when:**
- A best baseline model is selected using imbalance-aware metrics.

## Phase 5: Threshold Optimization
1. Sweep probability thresholds for top 1-2 models.
2. Select threshold by policy:
   - Max recall with minimum precision constraint, or
   - Max F1
3. Add confusion matrix at selected threshold.

**Done when:**
- A justified operating threshold is chosen (not just 0.5 default).

## Phase 6: Hyperparameter Tuning
1. Run `GridSearchCV` on the best model.
2. Use imbalance-aware scoring (`f1` or `average_precision`).
3. Compare tuned vs untuned test metrics.

**Done when:**
- Tuned model results are documented in `outputs/tuned_best_model_summary.csv`.

## Phase 7: Feature Importance and Interpretation
1. Export and review feature importance for supported models.
2. Highlight top predictive features.
3. Note interpretation limits due to PCA-transformed inputs.

**Done when:**
- Interpretation section includes meaningful and realistic conclusions.

## Phase 8: Robustness Checks
1. Add cross-validation summary for key model.
2. Check overfitting indicators.
3. Optionally test alternate seeds/SMOTE settings.

**Done when:**
- Results are stable and not split-dependent.

## Phase 9: Notebook Finalization
1. Polish notebook flow:
   - Problem
   - EDA
   - Methodology
   - Results
   - Tuning
   - Interpretation
   - Conclusion
2. Ensure all figures/tables include concise takeaways.
3. Verify full reproducibility from top to bottom.

**Done when:**
- Notebook is presentation-ready and grader-friendly.

## Phase 10: Final Report
1. Write report sections:
   - Abstract
   - Problem and motivation
   - Dataset and challenge
   - Methodology
   - Results and analysis
   - Limitations and future work
   - Conclusion
2. Include strongest result table and key plots.

**Done when:**
- Report narrative aligns with notebook outputs.

## Phase 11: Presentation
1. Prepare 8-12 slides:
   - Motivation
   - Data challenge
   - Pipeline
   - Model comparison
   - Best model and threshold
   - Feature insights
   - Limitations and future work
2. Prepare short demo walkthrough.

**Done when:**
- Deck supports a clear 7-10 minute presentation.

## Phase 12: API, Frontend, and Deployment
1. Build an inference API (FastAPI) with endpoints:
   - `GET /health`
   - `POST /predict` for single transaction prediction
   - `POST /predict-batch` for batch prediction from CSV/JSON
2. Export and version the final trained model and preprocessing pipeline.
3. Build a frontend dashboard (React or Streamlit) to:
   - Input transaction features
   - Display fraud probability, decision threshold, and prediction label
   - Show basic model details and usage notes
4. Containerize the API and frontend with Docker.
5. Deploy to cloud (for example Render, Railway, or Azure App Service/Container Apps).
6. Add production files:
   - `Dockerfile`
   - `docker-compose.yml`
   - `.env.example`
   - deployment instructions in `README.md`

**Done when:**
- Public demo URL is available.
- API is reachable and returns valid predictions.
- Frontend can call deployed API and show prediction results.
- Deployment and local run steps are fully documented.

## Working Sequence (Step-by-Step)
1. Execute baseline run on real dataset.
2. Build EDA section and insights.
3. Add threshold optimization and confusion matrix.
4. Finalize tuning and interpretation.
5. Produce final notebook, report, and presentation assets.
6. Build API, frontend dashboard, and deploy the end-to-end app.
