import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def top_tfidf_terms_for_input(model, text: str, top_k: int = 8):
    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]

    vec = pre.named_transformers_["text"]
    feature_names = vec.get_feature_names_out()
    text_vector = vec.transform([text])

    if not hasattr(clf, "coef_"):
        return []

    coef = clf.coef_.ravel()[: len(feature_names)]
    scores = text_vector.multiply(coef).toarray().ravel()
    if not np.any(scores):
        return []

    top_idx = np.argsort(scores)[::-1][:top_k]
    terms = []
    for idx in top_idx:
        if scores[idx] <= 0:
            continue
        terms.append((feature_names[idx], float(scores[idx])))
    return terms


def main():
    parser = argparse.ArgumentParser(description="Evaluate a text and show explainability signals.")
    parser.add_argument("--model", type=Path, default=Path("models/spam_model.joblib"), help="Path to trained model")
    parser.add_argument("--text", type=str, required=True, help="Email text to analyze")
    args = parser.parse_args()

    model = joblib.load(args.model)
    sample = pd.DataFrame({"email": [args.text]})

    pred = int(model.predict(sample)[0])
    label = "spam" if pred == 1 else "ham"

    if hasattr(model, "decision_function"):
        margin = float(model.decision_function(sample)[0])
        confidence = 1.0 / (1.0 + np.exp(-margin))
    elif hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(sample)[0][1])
    else:
        confidence = None

    print(f"Prediction: {label}")
    if confidence is not None:
        print(f"Spam confidence: {confidence:.4f}")

    terms = top_tfidf_terms_for_input(model, args.text)
    print("Top influential spam terms:")
    if not terms:
        print("  (No strong spam term contribution found)")
    else:
        for term, score in terms:
            print(f"  - {term}: {score:.4f}")


if __name__ == "__main__":
    main()
