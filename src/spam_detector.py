from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline

from src.spam_pipeline import (
    evaluate_model as _evaluate_model,
    load_data as _load_data,
    predict_email as _predict_email,
    train_model as _train_model,
)


def load_data_wrapper(file_path: str | Path = "data/sample_emails.csv") -> pd.DataFrame:
    """Backward-compatible wrapper to load data."""
    return _load_data(file_path)


def train_model_wrapper(
    data: pd.DataFrame,
    model_type: str = "logistic_regression",
) -> Tuple[Pipeline, pd.Series, pd.Series, Dict[str, Any]]:
    """Backward-compatible wrapper to train model."""
    return _train_model(data, model_type=model_type)


def evaluate_model_wrapper(model: Pipeline, X_test: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
    """Backward-compatible wrapper to evaluate model."""
    return _evaluate_model(model, X_test, y_test, reports_dir=Path("reports"))


def predict_email_wrapper(model: Pipeline, text: str) -> Dict[str, Any]:
    """Backward-compatible wrapper to predict email."""
    return _predict_email(model, text)


# Backward compatible names
load_data = load_data_wrapper
train_model = train_model_wrapper
evaluate_model = evaluate_model_wrapper
predict_email = predict_email_wrapper
