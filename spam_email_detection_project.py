from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.spam_pipeline import (
    compare_models,
    configure_logging,
    cross_validate_models,
    evaluate_model,
    load_data,
    load_model,
    predict_email,
    resolve_data_path,
    save_model,
    train_model,
)

logger = logging.getLogger(__name__)


def run_train(data_path: Path, model_path: Path, model_type: str, compare: bool, cv_folds: int) -> None:
    """Train model(s), optionally compare, and save best model."""
    data = load_data(resolve_data_path(data_path))

    if cv_folds > 1:
        cv_summary = cross_validate_models(data, cv_folds=cv_folds)
        for name, stats in cv_summary.items():
            logger.info("Cross-validation summary | %s | mean_f1=%.4f | std=%.4f", name, stats["cv_f1_mean"], stats["cv_f1_std"])

    if compare:
        model, best_name, X_test, y_test, metrics = compare_models(data)
        logger.info("Best model selected by comparison: %s", best_name)
    else:
        model, X_test, y_test, metrics = train_model(data, model_type=model_type)

    save_model(model, model_path)
    eval_metrics = evaluate_model(model, X_test, y_test, reports_dir=Path("reports"))

    logger.info("Precision: %.4f | Recall: %.4f | F1-score: %.4f", eval_metrics["precision"], eval_metrics["recall"], eval_metrics["f1"])
    logger.info("Classification report:\n%s", metrics["classification_report"])


def run_predict(model_path: Path, text: str) -> None:
    """Load model and run a single prediction."""
    model = load_model(model_path)
    result = predict_email(model, text)
    logger.info("Prediction: %s", result["label"])
    if result["confidence"] is not None:
        logger.info("Spam confidence: %.4f", result["confidence"])


def run_evaluate(model_path: Path, data_path: Path) -> None:
    """Evaluate saved model and write reports."""
    model = load_model(model_path)
    data = load_data(resolve_data_path(data_path))

    from sklearn.model_selection import train_test_split

    X = data["email"]
    y = data["label"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    metrics = evaluate_model(model, X_test, y_test, reports_dir=Path("reports"))

    logger.info("Precision: %.4f | Recall: %.4f | F1-score: %.4f", metrics["precision"], metrics["recall"], metrics["f1"])


def main() -> None:
    """CLI entry point for train, predict, and evaluate commands."""
    parser = argparse.ArgumentParser(description="Spam Email Detection using Machine Learning")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and save spam model")
    train_parser.add_argument("--data", type=Path, default=Path("data/sample_emails.csv"))
    train_parser.add_argument("--model-out", type=Path, default=Path("models/spam_model.joblib"))
    train_parser.add_argument(
        "--model-type",
        type=str,
        default="logistic_regression",
        choices=["naive_bayes", "logistic_regression"],
    )
    train_parser.add_argument("--compare-models", action="store_true", help="Compare Naive Bayes and Logistic Regression")
    train_parser.add_argument("--cv-folds", type=int, default=0, help="Optional cross-validation folds (0 disables CV)")

    predict_parser = subparsers.add_parser("predict", help="Predict spam/ham for one email")
    predict_parser.add_argument("--model", type=Path, default=Path("models/spam_model.joblib"))
    predict_parser.add_argument("--text", type=str, required=True)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate model and save reports")
    evaluate_parser.add_argument("--model", type=Path, default=Path("models/spam_model.joblib"))
    evaluate_parser.add_argument("--data", type=Path, default=Path("data/sample_emails.csv"))

    args = parser.parse_args()
    configure_logging()

    try:
        if args.command == "train":
            run_train(args.data, args.model_out, args.model_type, args.compare_models, args.cv_folds)
        elif args.command == "predict":
            run_predict(args.model, args.text)
        elif args.command == "evaluate":
            run_evaluate(args.model, args.data)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("Command failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
