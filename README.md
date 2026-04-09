# Credit Card Fraud Detection Using Classical ML with Imbalanced Learning

This project implements an end-to-end fraud detection workflow for highly imbalanced data using classical machine learning models.

## Implemented Baseline
- Data loading from the Kaggle credit card fraud CSV
- Stratified train/test split
- Optional SMOTE in training pipeline
- Four models:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - SVM
- Metrics:
  - Precision, Recall, F1, ROC-AUC, PR-AUC
- Curves:
  - ROC
  - Precision-Recall
- Hyperparameter tuning with GridSearchCV for the best baseline model

## Project Structure
- `src/pipeline.py`: Training, evaluation, plotting, and tuning functions
- `src/train.py`: CLI entrypoint to run full experiment
- `src/export_serving_model.py`: Export versioned model artifact for inference API
- `api/main.py`: FastAPI inference service (`/health`, `/predict`, `/predict-batch`)
- `frontend/app.py`: Streamlit dashboard for single and batch scoring
- `notebooks/fraud_detection_pipeline.ipynb`: Starter notebook workflow
- `outputs/`: Generated metrics, plots, and tuned model summary
- `artifacts/`: Exported model and metadata for serving

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
Place the dataset CSV in your workspace (for example: `data/creditcard.csv`) and run:

```bash
python -m src.train --data-path data/creditcard.csv --output-dir outputs
```

## Export Serving Model
After training, export model artifacts used by the API:

```bash
python -m src.export_serving_model --data-path data/creditcard.csv --artifacts-dir artifacts --version v1
```

## Run API and Frontend (Local)
Start API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Start frontend in a second terminal:

```bash
streamlit run frontend/app.py
```

Open:
- API docs: `http://localhost:8000/docs`
- Frontend dashboard: `http://localhost:8501`

## Run with Docker Compose
```bash
docker compose up --build
```

## Deployment
Detailed deployment steps are in `DEPLOYMENT_GUIDE.md`.

## Notes
- Ensure SMOTE is applied only on training folds via pipeline to avoid leakage.
- Best model is selected by `f1` in the current baseline; this can be changed to `average_precision` depending on your priority.
