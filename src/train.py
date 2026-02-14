import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC


def handcrafted_from_df(x: pd.DataFrame) -> np.ndarray:
    emails = x.iloc[:, 0].fillna("").astype(str)

    exclamation_count = emails.str.count(r"!").to_numpy(dtype=float)
    url_flag = emails.str.contains(r"https?://|www\\.", case=False, regex=True).astype(float).to_numpy()
    urgency_flag = emails.str.contains(
        r"urgent|immediately|final warning|act now|verify|suspended|locked|payment failed",
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
        "linear_svc": LinearSVC(class_weight="balanced"),
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


def main():
    parser = argparse.ArgumentParser(description="Train a unique spam email detector.")
    parser.add_argument("--data", type=Path, default=Path("data/sample_emails.csv"), help="Path to CSV with columns: email,label")
    parser.add_argument("--model-out", type=Path, default=Path("models/spam_model.joblib"), help="Output path for serialized model")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    required = {"email", "label"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Dataset missing required columns: {missing}")

    model = train_and_select_model(df)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)
    print(f"Saved model: {args.model_out}")


if __name__ == "__main__":
    main()
