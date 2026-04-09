from __future__ import annotations

import json
import io
import os
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP_ROOT = Path(__file__).resolve().parent.parent
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(APP_ROOT / "artifacts")))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ARTIFACTS_DIR / f"fraud_model_{MODEL_VERSION}.joblib")))
METADATA_PATH = Path(os.getenv("MODEL_METADATA_PATH", str(ARTIFACTS_DIR / f"fraud_model_{MODEL_VERSION}.json")))


class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Feature dictionary for one transaction")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


class PredictBatchRequest(BaseModel):
    records: Optional[List[Dict[str, float]]] = None
    csv_text: Optional[str] = None
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


app = FastAPI(title="Fraud Detection API", version="1.0.0")

model = None
metadata = {}
feature_order: List[str] = []
default_threshold = 0.5


@app.on_event("startup")
def _startup() -> None:
    global model, metadata, feature_order, default_threshold

    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError(
            "Model artifacts not found. Run `python -m src.export_serving_model --data-path data/creditcard.csv` first."
        )

    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    feature_order = metadata.get("feature_order", [])
    default_threshold = float(metadata.get("threshold", 0.5))


@app.get("/health")
def health() -> Dict[str, object]:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": metadata.get("version", MODEL_VERSION),
        "feature_count": len(feature_order),
        "default_threshold": default_threshold,
    }


@app.get("/model-info")
def model_info() -> Dict[str, object]:
    return metadata


def _validate_and_frame(records: List[Dict[str, float]]) -> pd.DataFrame:
    if not records:
        raise HTTPException(status_code=400, detail="No records provided.")

    missing = [col for col in feature_order if col not in records[0]]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required features: {missing}")

    df = pd.DataFrame(records)
    return df[feature_order]


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, object]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    threshold = req.threshold if req.threshold is not None else default_threshold
    frame = _validate_and_frame([req.features])

    prob = float(model.predict_proba(frame)[:, 1][0])
    pred = int(prob >= threshold)

    return {
        "probability": prob,
        "prediction": pred,
        "threshold": threshold,
        "label": "fraud" if pred == 1 else "non-fraud",
    }


@app.post("/predict-batch")
def predict_batch(req: PredictBatchRequest) -> Dict[str, object]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    threshold = req.threshold if req.threshold is not None else default_threshold

    if req.csv_text:
        frame = pd.read_csv(io.StringIO(req.csv_text))
        missing = [col for col in feature_order if col not in frame.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"CSV missing required features: {missing}")
        data = frame[feature_order]
    elif req.records is not None:
        data = _validate_and_frame(req.records)
    else:
        raise HTTPException(status_code=400, detail="Provide either `records` or `csv_text`.")

    probs = model.predict_proba(data)[:, 1]
    preds = (probs >= threshold).astype(int)

    results = [
        {
            "index": int(i),
            "probability": float(probs[i]),
            "prediction": int(preds[i]),
            "label": "fraud" if int(preds[i]) == 1 else "non-fraud",
        }
        for i in range(len(data))
    ]

    return {
        "count": len(results),
        "threshold": threshold,
        "results": results,
    }
