from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
TARGET_COLUMN = "Class"


def load_data(data_path: str | Path) -> pd.DataFrame:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    return pd.read_csv(data_path)


def split_data(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def get_models() -> Dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
        "svm": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def _needs_scaling(model_name: str) -> bool:
    return model_name in {"logistic_regression", "svm"}


def build_pipeline(model_name: str, model: object, use_smote: bool = True) -> ImbPipeline:
    steps = []
    if _needs_scaling(model_name):
        steps.append(("scaler", StandardScaler()))

    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))

    steps.append(("model", model))
    return ImbPipeline(steps=steps)


def evaluate_model(model: ImbPipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
    }


def plot_curves(
    model_name: str,
    model: ImbPipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=model_name)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"roc_{model_name}.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=model_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve: {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"pr_{model_name}.png", dpi=150)
    plt.close()


def _get_param_grid(model_name: str) -> Dict[str, list]:
    grids = {
        "logistic_regression": {
            "model__C": [0.1, 1.0, 10.0],
            "model__solver": ["liblinear", "lbfgs"],
        },
        "random_forest": {
            "model__n_estimators": [200, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
        },
        "gradient_boosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        },
        "svm": {
            "model__C": [1.0, 5.0],
            "model__gamma": ["scale", 0.1],
            "model__kernel": ["rbf"],
        },
    }
    return grids[model_name]


def _save_feature_importance(
    model_name: str,
    fitted_pipeline: ImbPipeline,
    feature_names: pd.Index,
    output_dir: Path,
) -> None:
    estimator = fitted_pipeline.named_steps["model"]

    importance_df = None
    if hasattr(estimator, "feature_importances_"):
        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": estimator.feature_importances_,
            }
        )
    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        if getattr(coef, "ndim", 1) > 1:
            coef = coef[0]
        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": coef,
            }
        )

    if importance_df is None:
        return

    importance_df["abs_importance"] = importance_df["importance"].abs()
    importance_df = importance_df.sort_values(by="abs_importance", ascending=False)
    importance_df.to_csv(output_dir / f"feature_importance_{model_name}.csv", index=False)


def run_experiment(data_path: str | Path, output_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df)

    rows = []
    trained_models: Dict[str, ImbPipeline] = {}

    for model_name, model in get_models().items():
        pipe = build_pipeline(model_name=model_name, model=model, use_smote=True)
        pipe.fit(X_train, y_train)

        metrics = evaluate_model(pipe, X_test, y_test)
        metrics["model"] = model_name
        rows.append(metrics)
        trained_models[model_name] = pipe

        _save_feature_importance(
            model_name=model_name,
            fitted_pipeline=pipe,
            feature_names=X_train.columns,
            output_dir=output_dir,
        )

        plot_curves(model_name, pipe, X_test, y_test, output_dir)

    results_df = pd.DataFrame(rows).sort_values(by="f1", ascending=False)
    results_df.to_csv(output_dir / "baseline_results.csv", index=False)

    best_model_name = results_df.iloc[0]["model"]
    best_model = get_models()[best_model_name]
    best_pipe = build_pipeline(best_model_name, best_model, use_smote=True)

    search = GridSearchCV(
        estimator=best_pipe,
        param_grid=_get_param_grid(best_model_name),
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    tuned_metrics = evaluate_model(search.best_estimator_, X_test, y_test)
    tuned_summary = {
        "best_model": best_model_name,
        "best_params": search.best_params_,
        "cv_best_score_f1": search.best_score_,
        "test_metrics": tuned_metrics,
    }

    pd.Series({
        "best_model": tuned_summary["best_model"],
        "best_params": str(tuned_summary["best_params"]),
        "cv_best_score_f1": tuned_summary["cv_best_score_f1"],
        "test_precision": tuned_summary["test_metrics"]["precision"],
        "test_recall": tuned_summary["test_metrics"]["recall"],
        "test_f1": tuned_summary["test_metrics"]["f1"],
        "test_roc_auc": tuned_summary["test_metrics"]["roc_auc"],
        "test_pr_auc": tuned_summary["test_metrics"]["pr_auc"],
    }).to_csv(output_dir / "tuned_best_model_summary.csv")

    return results_df
