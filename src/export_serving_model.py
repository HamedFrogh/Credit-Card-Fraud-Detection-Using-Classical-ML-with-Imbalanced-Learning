from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.pipeline import build_pipeline, get_models, load_data, split_data

DEFAULT_THRESHOLD = 0.48


def export_model(data_path: Path, artifacts_dir: Path, version: str) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    X_train, _, y_train, _ = split_data(df)

    model = get_models()["random_forest"]
    # Keep parameters aligned with best tuned configuration found in project outputs.
    model.set_params(n_estimators=200, max_depth=None, min_samples_split=2)

    pipe = build_pipeline("random_forest", model, use_smote=True)
    pipe.fit(X_train, y_train)

    model_path = artifacts_dir / f"fraud_model_{version}.joblib"
    meta_path = artifacts_dir / f"fraud_model_{version}.json"

    metadata = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": "random_forest",
        "threshold": DEFAULT_THRESHOLD,
        "target_column": "Class",
        "feature_order": list(X_train.columns),
        "notes": "Model trained with SMOTE pipeline for serving inference.",
    }

    joblib.dump(pipe, model_path)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {meta_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained serving model artifact.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to creditcard.csv")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Output artifacts folder")
    parser.add_argument("--version", type=str, default="v1", help="Model version suffix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_model(
        data_path=Path(args.data_path),
        artifacts_dir=Path(args.artifacts_dir),
        version=args.version,
    )


if __name__ == "__main__":
    main()
