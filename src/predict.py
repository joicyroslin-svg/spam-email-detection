import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def spam_probability_from_decision(decision_value: float) -> float:
    # Map margin score to pseudo-probability for consistent CLI output.
    return 1.0 / (1.0 + np.exp(-decision_value))


def main():
    parser = argparse.ArgumentParser(description="Predict whether an email is spam.")
    parser.add_argument("--model", type=Path, default=Path("models/spam_model.joblib"), help="Path to trained model")
    parser.add_argument("--text", type=str, required=True, help="Email text to classify")
    args = parser.parse_args()

    model = joblib.load(args.model)
    sample = pd.DataFrame({"email": [args.text]})

    pred = int(model.predict(sample)[0])
    label = "spam" if pred == 1 else "ham"

    confidence = None
    clf = model.named_steps.get("classifier")
    if hasattr(clf, "predict_proba"):
        confidence = float(model.predict_proba(sample)[0][1])
    elif hasattr(model, "decision_function"):
        margin = float(model.decision_function(sample)[0])
        confidence = spam_probability_from_decision(margin)

    print(f"Prediction: {label}")
    if confidence is not None:
        print(f"Spam confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()