import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
import warnings


DEFAULT_DATA = Path("data/sample_emails.csv")
DEFAULT_MODEL = Path("models/spam_model.joblib")

RISK_PATTERNS = {
    "credential_theft": r"password|otp|pin|verify your account|login now|confirm identity|kyc",
    "financial_bait": r"won|lottery|reward|cashback|gift card|free money|guaranteed profits|instant loan",
    "urgent_pressure": r"urgent|immediately|act now|final warning|24 hours|suspended|locked|termination",
    "payment_fraud": r"payment failed|re-enter card|pay fee|wire transfer|settlement pending",
    "suspicious_link_language": r"click now|open this link|secure form|verify through this link",
}


def extract_risk_signals(text: str):
    lowered = (text or "").lower()
    signals = {}
    for name, pattern in RISK_PATTERNS.items():
        matches = re.findall(pattern, lowered, flags=re.IGNORECASE)
        if matches:
            signals[name] = sorted(set(matches))
    return signals


def handcrafted_from_df(x: pd.DataFrame) -> np.ndarray:
    emails = x.iloc[:, 0].fillna("").astype(str)

    exclamation_count = emails.str.count(r"!").to_numpy(dtype=float)
    url_flag = emails.str.contains(r"https?://|www\\.|bit\\.ly|tinyurl", case=False, regex=True).astype(float).to_numpy()
    urgency_flag = emails.str.contains(
        r"urgent|immediately|final warning|act now|verify|suspended|locked|payment failed|termination",
        case=False,
        regex=True,
    ).astype(float).to_numpy()
    money_flag = emails.str.contains(
        r"won|lottery|cash|reward|bonus|earn|loan|free money|gift card",
        case=False,
        regex=True,
    ).astype(float).to_numpy()

    upper_ratio = []
    digit_ratio = []
    char_length = []

    for s in emails:
        alpha = sum(1 for c in s if c.isalpha())
        upper = sum(1 for c in s if c.isupper())
        digits = sum(1 for c in s if c.isdigit())

        upper_ratio.append((upper / alpha) if alpha else 0.0)
        digit_ratio.append((digits / len(s)) if s else 0.0)
        char_length.append(float(len(s)))

    return np.column_stack(
        [
            exclamation_count,
            url_flag,
            urgency_flag,
            money_flag,
            np.array(upper_ratio, dtype=float),
            np.array(digit_ratio, dtype=float),
            np.array(char_length, dtype=float),
        ]
    )


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_df=0.95,
                    strip_accents="unicode",
                    lowercase=True,
                    sublinear_tf=True,
                ),
                "email",
            ),
            (
                "meta",
                FunctionTransformer(handcrafted_from_df, validate=False),
                ["email"],
            ),
        ],
        remainder="drop",
    )


def train_and_select_model(df: pd.DataFrame):
    X = df[["email"]]
    y = df["label"].map({"ham": 0, "spam": 1})

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
        "linear_svc": LinearSVC(class_weight="balanced", max_iter=10000, tol=1e-4),
    }

    best_name = None
    best_score = -1.0
    best_pipeline = None

    for name, estimator in candidates.items():
        model = Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("classifier", estimator),
            ]
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        score = f1_score(y_val, y_pred)

        print(f"\\nModel: {name}")
        print(f"F1 score: {score:.4f}")
        print(classification_report(y_val, y_pred, target_names=["ham", "spam"]))

        if score > best_score:
            best_score = score
            best_name = name
            best_pipeline = model

    print(f"Best model selected: {best_name} (F1={best_score:.4f})")
    return best_pipeline


def spam_probability(model, sample: pd.DataFrame):
    clf = model.named_steps.get("classifier")
    if hasattr(clf, "predict_proba"):
        return float(model.predict_proba(sample)[0][1])
    if hasattr(model, "decision_function"):
        margin = float(model.decision_function(sample)[0])
        return float(1.0 / (1.0 + np.exp(-margin)))
    return None


def top_tfidf_terms_for_input(model, text: str, top_k: int = 8):
    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]
    if not hasattr(clf, "coef_"):
        return []

    vec = pre.named_transformers_["text"]
    feature_names = vec.get_feature_names_out()
    text_vector = vec.transform([text])

    coef = clf.coef_.ravel()[: len(feature_names)]
    scores = text_vector.multiply(coef).toarray().ravel()

    top_idx = np.argsort(scores)[::-1][:top_k]
    terms = []
    for idx in top_idx:
        if scores[idx] <= 0:
            continue
        terms.append((feature_names[idx], float(scores[idx])))
    return terms


def risk_score_from_signals(signals):
    weights = {
        "credential_theft": 30,
        "payment_fraud": 25,
        "urgent_pressure": 20,
        "financial_bait": 20,
        "suspicious_link_language": 15,
    }
    score = sum(weights.get(k, 0) for k in signals.keys())
    return min(score, 100)


def run_train(data_path: Path, model_out: Path):
    df = pd.read_csv(data_path)
    required = {"email", "label"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Dataset missing required columns: {missing}")

    model = train_and_select_model(df)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    print(f"Saved model: {model_out}")


def run_predict(model_path: Path, text: str):
    model = joblib.load(model_path)
    sample = pd.DataFrame({"email": [text]})
    pred = int(model.predict(sample)[0])
    label = "spam" if pred == 1 else "ham"

    print(f"Prediction: {label}")
    conf = spam_probability(model, sample)
    if conf is not None:
        print(f"Spam confidence: {conf:.4f}")


def run_explain(model_path: Path, text: str):
    model = joblib.load(model_path)
    sample = pd.DataFrame({"email": [text]})
    pred = int(model.predict(sample)[0])
    label = "spam" if pred == 1 else "ham"

    print(f"Prediction: {label}")
    conf = spam_probability(model, sample)
    if conf is not None:
        print(f"Spam confidence: {conf:.4f}")

    terms = top_tfidf_terms_for_input(model, text)
    print("Top influential spam terms:")
    if not terms:
        print("  (No strong spam term contribution found)")
    else:
        for term, score in terms:
            print(f"  - {term}: {score:.4f}")


def run_analyze(model_path: Path, text: str):
    model = joblib.load(model_path)
    sample = pd.DataFrame({"email": [text]})

    pred = int(model.predict(sample)[0])
    label = "spam" if pred == 1 else "ham"
    conf = spam_probability(model, sample)
    signals = extract_risk_signals(text)
    risk_score = risk_score_from_signals(signals)

    print(f"Prediction: {label}")
    if conf is not None:
        print(f"Spam confidence: {conf:.4f}")
    print(f"Rule-based risk score: {risk_score}/100")

    if not signals:
        print("Matched risk signals: none")
    else:
        print("Matched risk signals:")
        for key, values in signals.items():
            print(f"  - {key}: {', '.join(values)}")

    if label == "spam" or risk_score >= 40:
        print("Action: Move to spam/quarantine and avoid clicking links.")
    else:
        print("Action: Looks relatively safe, but still verify sender identity.")


def main():
    parser = argparse.ArgumentParser(description="Unique Spam Email Detection - Single File Project")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train model and save best classifier")
    train_parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    train_parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)

    predict_parser = subparsers.add_parser("predict", help="Predict spam/ham for one email text")
    predict_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    predict_parser.add_argument("--text", type=str, required=True)

    explain_parser = subparsers.add_parser("explain", help="Predict and show top influential spam terms")
    explain_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    explain_parser.add_argument("--text", type=str, required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Predict with risk signals and safety action")
    analyze_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    analyze_parser.add_argument("--text", type=str, required=True)

    args = parser.parse_args()

    if args.command == "train":
        run_train(args.data, args.model_out)
    elif args.command == "predict":
        run_predict(args.model, args.text)
    elif args.command == "explain":
        run_explain(args.model, args.text)
    elif args.command == "analyze":
        run_analyze(args.model, args.text)


if __name__ == "__main__":
    main()
