from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from src.spam_pipeline import (
    compare_models,
    configure_logging,
    cross_validate_models,
    evaluate_benchmark_datasets,
    evaluate_model,
    load_data,
    load_model,
    predict_email,
    resolve_data_path,
    save_explainability_report,
    save_model,
    save_model_metadata,
    train_model,
    tune_hyperparameters_randomized,
)

logger = logging.getLogger(__name__)


def run_train(
    data_path: Path,
    model_path: Path,
    model_type: str,
    compare: bool,
    cv_folds: int,
    tune_hyperparams: bool,
    n_iter: int,
    calibrate: bool,
    benchmark_data: Path | None,
) -> None:
    """Train model(s), optionally tune/calibrate, evaluate, and save artifacts."""
    resolved_data = resolve_data_path(data_path)
    data = load_data(resolved_data)
    best_params: dict[str, Any] = {}

    if cv_folds > 1:
        cv_summary = cross_validate_models(data, cv_folds=cv_folds)
        for name, stats in cv_summary.items():
            logger.info(
                "Cross-validation summary | %s | mean_f1=%.4f | std=%.4f",
                name,
                stats["cv_f1_mean"],
                stats["cv_f1_std"],
            )

    if compare:
        model, best_name, X_test, y_test, metrics = compare_models(data, calibrate=calibrate)
        logger.info("Best model selected by comparison: %s", best_name)
        model_type = best_name
    else:
        model, X_test, y_test, metrics = train_model(data, model_type=model_type, calibrate=calibrate)

    tuning_path = None
    if tune_hyperparams:
        tuned_model, best_params, tuned_cv_f1, tuning_path = tune_hyperparameters_randomized(
            data,
            model_type=model_type,
            cv_folds=max(cv_folds, 3),
            n_iter=n_iter,
            reports_dir=Path("reports"),
        )
        model = tuned_model
        metrics["tuned_cv_f1"] = tuned_cv_f1
        logger.info("Using tuned %s model with CV F1=%.4f", model_type, tuned_cv_f1)

    save_model(model, model_path)
    eval_metrics = evaluate_model(model, X_test, y_test, reports_dir=Path("reports"))
    explainability_path = save_explainability_report(model, str(X_test.iloc[0]), reports_dir=Path("reports"))

    benchmark_paths = [resolved_data]
    default_benchmark = Path("data/benchmark_emails.csv")
    if benchmark_data and benchmark_data.exists():
        benchmark_paths.append(benchmark_data)
    elif default_benchmark.exists() and default_benchmark != resolved_data:
        benchmark_paths.append(default_benchmark)
    benchmark_path = evaluate_benchmark_datasets(model, benchmark_paths, reports_dir=Path("reports"))

    artifact_map = {
        "evaluation": str(eval_metrics["evaluation_path"]),
        "confusion_matrix": str(eval_metrics["confusion_matrix_path"]),
        "roc_curve": str(eval_metrics["roc_curve_path"]),
        "pr_curve": str(eval_metrics["pr_curve_path"]),
        "explainability": str(explainability_path),
        "benchmark_results": str(benchmark_path),
    }
    if tuning_path is not None:
        artifact_map["tuning_results"] = str(tuning_path)

    save_model_metadata(
        model,
        eval_metrics,
        resolved_data,
        model_path,
        reports_dir=Path("reports"),
        best_params=best_params,
        extra_artifacts=artifact_map,
    )

    logger.info(
        "Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1-score: %.4f",
        eval_metrics["accuracy"],
        eval_metrics["precision"],
        eval_metrics["recall"],
        eval_metrics["f1"],
    )
    logger.info("Classification report:\n%s", metrics["classification_report"])


def run_predict(model_path: Path, text: str) -> None:
    """Load model and run a single prediction."""
    model = load_model(model_path)
    result = predict_email(model, text)
    logger.info("Prediction: %s", result["label"])
    if result["confidence"] is not None:
        logger.info("Spam confidence: %.4f", result["confidence"])
    if result["top_terms"]:
        logger.info("Top terms: %s", ", ".join(term for term, _ in result["top_terms"][:5]))


def run_evaluate(model_path: Path, data_path: Path) -> None:
    """Evaluate saved model and write reports."""
    model = load_model(model_path)
    data = load_data(resolve_data_path(data_path))

    from sklearn.model_selection import train_test_split

    X = data["email"]
    y = data["label"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    metrics = evaluate_model(model, X_test, y_test, reports_dir=Path("reports"))

    logger.info(
        "Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1-score: %.4f | ROC-AUC: %.4f | PR-AUC: %.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["roc_auc"],
        metrics["pr_auc"],
    )


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
    train_parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Compare Naive Bayes and Logistic Regression",
    )
    train_parser.add_argument("--cv-folds", type=int, default=0, help="Optional cross-validation folds (0 disables CV)")
    train_parser.add_argument("--tune-hyperparams", action="store_true", help="Run RandomizedSearchCV tuning")
    train_parser.add_argument("--n-iter", type=int, default=10, help="RandomizedSearchCV iterations")
    train_parser.add_argument("--calibrate", action="store_true", help="Calibrate probability outputs")
    train_parser.add_argument(
        "--benchmark-data",
        type=Path,
        default=Path("data/benchmark_emails.csv"),
        help="Optional benchmark dataset",
    )

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
            run_train(
                args.data,
                args.model_out,
                args.model_type,
                args.compare_models,
                args.cv_folds,
                args.tune_hyperparams,
                args.n_iter,
                args.calibrate,
                args.benchmark_data,
            )
        elif args.command == "predict":
            run_predict(args.model, args.text)
        elif args.command == "evaluate":
            run_evaluate(args.model, args.data)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("Command failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
