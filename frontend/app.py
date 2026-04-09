from __future__ import annotations

import io
import os
import time
from typing import Dict

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Credit Card Fraud Detection Dashboard")
st.caption("Classical ML + Imbalanced Learning Inference UI")


def get_model_info() -> Dict[str, object]:
    last_exc: Exception | None = None
    # Render free tier may cold-start; allow a few retries.
    for attempt in range(3):
        try:
            resp = requests.get(f"{API_BASE_URL}/model-info", timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(3)

    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=15)
        resp.raise_for_status()
        st.warning("API is reachable but model-info is timing out. Retry in a few seconds.")
        return {}
    except Exception as exc:
        if last_exc is not None:
            st.error(f"API unavailable: {last_exc}")
        else:
            st.error(f"API unavailable: {exc}")
        st.info(
            "Check `API_BASE_URL` in Render frontend environment settings and verify `/health` on the API service URL."
        )
        return {}


info = get_model_info()
if not info:
    st.stop()

feature_order = info.get("feature_order", [])
default_threshold = float(info.get("threshold", 0.5))

st.sidebar.header("API Settings")
st.sidebar.write(f"Base URL: `{API_BASE_URL}`")
threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, default_threshold, 0.01)

with st.expander("Model Metadata", expanded=False):
    st.json(info)

single_tab, batch_tab = st.tabs(["Single Prediction", "Batch Prediction"])

with single_tab:
    st.subheader("Single Transaction")
    cols = st.columns(3)

    values = {}
    for idx, name in enumerate(feature_order):
        col = cols[idx % 3]
        default_val = 0.0
        if name == "Amount":
            default_val = 50.0
        values[name] = col.number_input(name, value=float(default_val), step=0.01)

    if st.button("Predict Single", type="primary"):
        payload = {"features": values, "threshold": threshold}
        resp = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=15)
        if resp.ok:
            result = resp.json()
            st.success(
                f"Prediction: {result['label']} | Probability: {result['probability']:.4f} | Threshold: {result['threshold']:.2f}"
            )
            st.json(result)
        else:
            st.error(f"Prediction failed: {resp.text}")

with batch_tab:
    st.subheader("Batch CSV Prediction")
    st.write("Upload CSV containing all required model features.")
    file = st.file_uploader("CSV file", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)
        st.write("Preview", df.head())
        csv_text = df.to_csv(index=False)

        if st.button("Predict Batch", type="primary"):
            payload = {"csv_text": csv_text, "threshold": threshold}
            resp = requests.post(f"{API_BASE_URL}/predict-batch", json=payload, timeout=30)
            if resp.ok:
                result = resp.json()
                out_df = pd.DataFrame(result["results"])
                st.success(f"Scored {result['count']} rows")
                st.dataframe(out_df)

                csv_buffer = io.StringIO()
                out_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download Predictions CSV",
                    data=csv_buffer.getvalue(),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
            else:
                st.error(f"Batch prediction failed: {resp.text}")
