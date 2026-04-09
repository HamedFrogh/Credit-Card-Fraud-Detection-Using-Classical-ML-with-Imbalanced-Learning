# Project Progress Tracker

Track completion against `PROJECT_ROADMAP.md`.

## Phase 1: Setup and Data Readiness
- [x] Confirm Python environment and dependencies.
- [x] Place dataset at `data/creditcard.csv`.
- [x] Run baseline pipeline once end-to-end.

Notes:
- Environment `.venv` is configured.
- Core dependencies from `requirements.txt` are installed.
- Dependency import check passed (`env_ok`).
- Baseline run completed successfully; outputs generated in `outputs/`.

## Phase 2: EDA and Data Understanding
- [x] Add class imbalance visualization.
- [x] Analyze `Amount` and `Time` distributions.
- [x] Add correlation analysis with fraud class.
- [x] Write short insights for each EDA plot.

Notes:
- Fraud rate confirmed at 0.1727% (492 fraud transactions out of 284,807).
- `Amount` is strongly right-skewed with heavy outliers.
- Top absolute correlations with `Class`: V17, V14, V12, V10, V16, V3, V7, V11.
- EDA section added and executed in `notebooks/fraud_detection_pipeline.ipynb`.

## Phase 3: Preprocessing and Evaluation Protocol
- [x] Use stratified train/test split.
- [x] Apply scaling only for LR and SVM.
- [x] Apply SMOTE only on training folds to avoid leakage.
- [x] Add reproducibility details (seed, package versions).

## Phase 4: Baseline Model Benchmarking
- [x] Implement training for Logistic Regression.
- [x] Implement training for Random Forest.
- [x] Implement training for Gradient Boosting.
- [x] Implement training for SVM.
- [x] Implement evaluation metrics (Precision, Recall, F1, ROC-AUC, PR-AUC).
- [x] Save ROC and PR curves.
- [x] Select best baseline model on real data.

Notes:
- Best baseline model by F1: `random_forest` (F1 = 0.8367, Precision = 0.8367, Recall = 0.8367).

## Phase 5: Threshold Optimization
- [x] Sweep probability thresholds for top model(s).
- [x] Select threshold by policy.
- [x] Add confusion matrix at selected threshold.

Notes:
- Model used: `random_forest`.
- Policy used: maximize recall with precision >= 0.80.
- Selected threshold: `0.48`.
- Metrics at selected threshold: Precision = `0.8333`, Recall = `0.8673`, F1 = `0.8500`.
- Confusion matrix at t=0.48: TN=56846, FP=18, FN=13, TP=85.
- Threshold sweep table saved to `outputs/threshold_optimization_random_forest.csv`.

## Phase 6: Hyperparameter Tuning
- [x] Add `GridSearchCV` flow for best model.
- [x] Run tuning on real data.
- [x] Compare tuned vs untuned metrics.

Notes:
- Tuned best model: `random_forest`.
- Test F1 after tuning: 0.8265 (slightly below baseline 0.8367).
- Tuning artifacts are saved in `outputs/tuned_best_model_summary.csv`.

## Phase 7: Feature Importance and Interpretation
- [x] Add feature importance export for supported models.
- [x] Review top predictive features from real-data run.
- [x] Add interpretation caveats in notebook/report.

Notes:
- Feature importance files generated for Logistic Regression, Random Forest, and Gradient Boosting.
- Interpretation caveats documented in notebook (PCA interpretability limits, correlation vs causation, threshold sensitivity, SMOTE caveat, and drift monitoring).

## Phase 8: Robustness Checks
- [x] Add cross-validation summary for key model.
- [x] Check overfitting indicators.
- [x] Test alternate seeds/SMOTE settings (optional).

Notes:
- Cross-validation (3-fold) for `random_forest` completed:
	- Precision mean/std: 0.8793 / 0.0110
	- Recall mean/std: 0.8096 / 0.0312
	- F1 mean/std: 0.8425 / 0.0120
	- ROC-AUC mean/std: 0.9758 / 0.0085
	- PR-AUC mean/std: 0.8414 / 0.0131
- Overfitting diagnostic (default threshold) shows a train-test gap:
	- Train F1: 1.0000
	- Test F1: 0.8265
- Alternate split seeds (42, 7, 2026) show test F1 in range 0.8265 to 0.8691.
- Robustness artifacts saved:
	- `outputs/robustness_cv_summary_random_forest.csv`
	- `outputs/robustness_train_test_overfit_random_forest.csv`
	- `outputs/robustness_seed_sensitivity_random_forest.csv`

## Phase 9: Notebook Finalization
- [x] Polish notebook narrative flow.
- [x] Add concise takeaways for each table/plot.
- [x] Verify full top-to-bottom reproducibility.

Notes:
- Notebook intro and flow were polished with clear section ordering.
- Added concise takeaways for baseline results, overfitting check, and final conclusion.
- Full code-cell execution pass completed successfully in order, including end-to-end retraining cell.
- Reproducibility outputs remain consistent with earlier results (best baseline model, threshold, and robustness metrics).

## Phase 10: Final Report
- [x] Draft full report sections.
- [x] Insert key plots and strongest result table.
- [x] Finalize references and conclusion.

Notes:
- Final report created at `FINAL_PROJECT_REPORT.md`.
- Report includes abstract, methodology, EDA summary, model comparison table, tuning, threshold optimization, robustness checks, limitations, and conclusion.
- Report references generated artifacts in `outputs/` and includes academic/tool references.

## Phase 11: Presentation
- [x] Prepare 8-12 slide deck.
- [x] Prepare short demo walkthrough.

Notes:
- Slide-by-slide presentation prepared in `PRESENTATION_OUTLINE.md` (10-slide structure with speaker notes).
- Live demo script prepared in `PRESENTATION_DEMO_WALKTHROUGH.md` (2-3 minute flow and backup Q&A points).

## Phase 12: API, Frontend, and Deployment
- [x] Build FastAPI inference endpoints (`/health`, `/predict`, `/predict-batch`).
- [x] Export/version final model for serving.
- [x] Build frontend dashboard.
- [x] Containerize API and frontend.
- [ ] Deploy and publish demo URL.
- [x] Document run/deployment instructions.

Notes:
- API implemented in `api/main.py` with endpoints: `/health`, `/model-info`, `/predict`, `/predict-batch`.
- Serving model export implemented in `src/export_serving_model.py`; artifacts generated in `artifacts/`.
- Frontend dashboard implemented in `frontend/app.py` for single and batch predictions.
- Containerization added: `api/Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml`, `.env.example`.
- Local container run validated with `sudo docker compose up --build`.
- Deployment docs created in `DEPLOYMENT_GUIDE.md` and `README.md`.
- Render blueprint config prepared in `render.yaml` for two-service deploy.
- API smoke test passed: local `/health` returned model-loaded status.
- Public demo URL is pending because actual cloud deployment requires user-selected provider account and credentials.
