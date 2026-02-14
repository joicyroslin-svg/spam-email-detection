from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.spam_pipeline import (
    configure_logging,
    evaluate_model,
    load_data,
    load_model,
    resolve_data_path,
)

logger = logging.getLogger(__name__)


def run_evaluation(model_path: Path, data_path: Path, reports_dir: Path) -> None:
    """Evaluate a trained model on dataset split and save reports."""
    model = load_model(model_path)
    data = load_data(resolve_data_path(data_path))

    # Use deterministic split from training helper by reusing train_test_split semantics.
    X = data["email"]
    y = data["label"]

    from sklearn.model_selection import train_test_split

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    metrics = evaluate_model(model, X_test, y_test, reports_dir=reports_dir)

    logger.info(
        "Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1-score: %.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )
    logger.info("Evaluation report saved at: %s", metrics["evaluation_path"])
    logger.info("Confusion matrix image saved at: %s", metrics["confusion_matrix_path"])


def main() -> None:
    """CLI entry point for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate spam detection model")
    parser.add_argument("--model", type=Path, default=Path("models/spam_model.joblib"), help="Path to trained model")
    parser.add_argument("--data", type=Path, default=Path("data/sample_emails.csv"), help="Path to dataset")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"), help="Output directory for reports")
    args = parser.parse_args()

    configure_logging()

    try:
        run_evaluation(args.model, args.data, args.reports_dir)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Evaluation failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
