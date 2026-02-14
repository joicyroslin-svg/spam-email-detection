from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.spam_pipeline import configure_logging, load_model, predict_email

logger = logging.getLogger(__name__)


def run_predict(model_path: Path, text: str) -> None:
    """Load model and predict a single email text."""
    model = load_model(model_path)
    result = predict_email(model, text)

    logger.info("Prediction: %s", result["label"])
    if result["confidence"] is not None:
        logger.info("Spam confidence: %.4f", result["confidence"])


def main() -> None:
    """CLI entry point for single-email prediction."""
    parser = argparse.ArgumentParser(description="Predict spam or ham for an email")
    parser.add_argument("--model", type=Path, default=Path("models/spam_model.joblib"), help="Path to trained model")
    parser.add_argument("--text", type=str, required=True, help="Email text")
    args = parser.parse_args()

    configure_logging()

    try:
        run_predict(args.model, args.text)
    except FileNotFoundError as exc:
        logger.error("Prediction failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
