from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.spam_pipeline import (
    compare_models,
    configure_logging,
    cross_validate_models,
    load_data,
    resolve_data_path,
    save_model,
    train_model,
)

logger = logging.getLogger(__name__)


def run_train(data_path: Path, model_out: Path, model_type: str, compare: bool, cv_folds: int) -> None:
    """Train model(s), select best model, and save to disk."""
    data = load_data(resolve_data_path(data_path))

    if cv_folds > 1:
        cv_summary = cross_validate_models(data, cv_folds=cv_folds)
        for name, stats in cv_summary.items():
            logger.info("Cross-validation summary | %s | mean_f1=%.4f | std=%.4f", name, stats["cv_f1_mean"], stats["cv_f1_std"])

    if compare:
        model, best_name, _, _, metrics = compare_models(data)
        logger.info("Model comparison enabled. Selected: %s", best_name)
    else:
        model, _, _, metrics = train_model(data, model_type=model_type)

    save_model(model, model_out)
    logger.info(
        "Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1-score: %.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )
    logger.info("Classification report:\n%s", metrics["classification_report"])


def main() -> None:
    """CLI entry point for model training."""
    parser = argparse.ArgumentParser(description="Train spam detection model")
    parser.add_argument("--data", type=Path, default=Path("data/sample_emails.csv"), help="Dataset path")
    parser.add_argument("--model-out", type=Path, default=Path("models/spam_model.joblib"), help="Output model path")
    parser.add_argument(
        "--model-type",
        choices=["naive_bayes", "logistic_regression"],
        default="logistic_regression",
        help="Model used when comparison is disabled",
    )
    parser.add_argument("--compare-models", action="store_true", help="Compare Naive Bayes and Logistic Regression")
    parser.add_argument("--cv-folds", type=int, default=0, help="Optional cross-validation folds (0 disables CV)")

    args = parser.parse_args()
    configure_logging()

    try:
        run_train(args.data, args.model_out, args.model_type, args.compare_models, args.cv_folds)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("Training failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
