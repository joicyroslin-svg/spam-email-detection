from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from src.spam_detector import evaluate_model, load_data, predict_email, train_model

DEFAULT_DATASET = Path("dataset/sample_emails.csv")
DEFAULT_MODEL = Path("model/spam_model.pkl")


def run_train(data_path: Path, model_path: Path, model_type: str) -> None:
    """Train the model, evaluate it, and save it to disk."""
    # Load and validate dataset.
    data = load_data(data_path)

    # Train selected model and get test split for evaluation.
    model, X_test, y_test = train_model(data, model_type=model_type)

    # Evaluate trained model.
    metrics = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("Classification Report:")
    print(metrics["classification_report"])
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])

    # Save trained model.
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to: {model_path}")


def run_predict(model_path: Path, text: str) -> None:
    """Load saved model and predict a single email."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    label = predict_email(model, text)
    print(f"Prediction: {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Spam Email Detection using Machine Learning")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and save a spam detection model")
    train_parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    train_parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    train_parser.add_argument(
        "--model-type",
        type=str,
        default="naive_bayes",
        choices=["naive_bayes", "logistic_regression"],
        help="Choose model type",
    )

    predict_parser = subparsers.add_parser("predict", help="Predict spam/ham for email text")
    predict_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    predict_parser.add_argument("--text", type=str, required=True)

    args = parser.parse_args()

    if args.command == "train":
        run_train(args.data, args.model_out, args.model_type)
    elif args.command == "predict":
        run_predict(args.model, args.text)


if __name__ == "__main__":
    main()

    # Example prediction code:
    # 1) Train first:
    #    py spam_email_detection_project.py train --model-type logistic_regression
    # 2) Predict:
    #    py spam_email_detection_project.py predict --text "Congratulations! You won a free iPhone. Click now!"
