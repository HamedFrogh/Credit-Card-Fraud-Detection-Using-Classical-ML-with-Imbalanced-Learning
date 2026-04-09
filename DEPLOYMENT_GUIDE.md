# Deployment Guide (API + Frontend)

## 1. Export Serving Model
Run once after training outputs are ready:

```bash
python -m src.export_serving_model --data-path data/creditcard.csv --artifacts-dir artifacts --version v1
```

This creates:
- `artifacts/fraud_model_v1.joblib`
- `artifacts/fraud_model_v1.json`

## 2. Run Locally (Non-Docker)
### Terminal A: API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Terminal B: Frontend
```bash
streamlit run frontend/app.py
```

Open:
- API docs: `http://localhost:8000/docs`
- Frontend: `http://localhost:8501`

## 3. Run with Docker Compose
```bash
docker compose up --build
```

Open:
- API docs: `http://localhost:8000/docs`
- Frontend: `http://localhost:8501`

## 4. API Endpoints
- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /predict-batch`

### `POST /predict` example
```json
{
  "features": {
    "Time": 10000,
    "V1": 0.1,
    "V2": -0.2,
    "V3": 0.05,
    "V4": 0.1,
    "V5": -0.1,
    "V6": 0.0,
    "V7": 0.1,
    "V8": -0.1,
    "V9": 0.0,
    "V10": 0.2,
    "V11": -0.2,
    "V12": 0.0,
    "V13": 0.1,
    "V14": -0.1,
    "V15": 0.0,
    "V16": 0.05,
    "V17": -0.05,
    "V18": 0.03,
    "V19": 0.02,
    "V20": -0.01,
    "V21": 0.0,
    "V22": 0.0,
    "V23": 0.0,
    "V24": 0.0,
    "V25": 0.0,
    "V26": 0.0,
    "V27": 0.0,
    "V28": 0.0,
    "Amount": 45.5
  },
  "threshold": 0.48
}
```

## 5. Cloud Deployment Options
## Option A: Render
- Recommended method: deploy both services with blueprint file `render.yaml`.

### Render Quickstart
1. Push this project to GitHub.
2. In Render dashboard, choose `New +` -> `Blueprint`.
3. Select your repository and deploy using `render.yaml`.
4. After first deployment, verify API URL:
  - `https://fraud-api.onrender.com/health`
5. Verify frontend URL:
  - `https://fraud-frontend.onrender.com`

If your Render service names differ, update frontend env var `API_BASE_URL` in Render settings to your actual API URL, then redeploy frontend.

## Option B: Railway
- Create two services (api + frontend), each with corresponding Dockerfile.
- Set env vars for API model paths and frontend API URL.

## Option C: Azure App Service / Container Apps
- Push images to a registry.
- Deploy API and frontend as separate container apps.
- Configure `API_BASE_URL` and model artifact env vars.

## 6. Required Environment Variables
- `MODEL_VERSION` (default `v1`)
- `ARTIFACTS_DIR` (default `artifacts`)
- `MODEL_PATH`
- `MODEL_METADATA_PATH`
- `API_BASE_URL` (frontend only)

## 7. Production Notes
- Add authentication for public endpoints.
- Add request logging and monitoring.
- Re-export model artifacts after retraining.
- Recalibrate threshold based on fraud operations feedback.

## 8. Repository Files for Deployment
- `render.yaml`
- `api/Dockerfile`
- `frontend/Dockerfile`
- `docker-compose.yml`
