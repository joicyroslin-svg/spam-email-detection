from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def load_data(file_path: str | Path = "dataset/sample_emails.csv") -> pd.DataFrame:
    """Load the dataset and validate required columns."""
    df = pd.read_csv(file_path)
    required_columns = {"email", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df[["email", "label"]].dropna()


def preprocess_text(text: str) -> str:
    """Normalize text for vectorization by lowercasing and removing noise."""
    text = str(text).lower()
    text = re.sub(r"https?://\\S+|www\\.\\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\\d+", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def _build_pipeline(model_type: str = "naive_bayes") -> Pipeline:
    """Build a text classification pipeline with TF-IDF and chosen classifier."""
    model_type = model_type.lower()

    # Step 1: Convert text into TF-IDF numeric features.
    vectorizer = TfidfVectorizer(preprocessor=preprocess_text, stop_words="english")

    # Step 2: Choose a classifier requested by the caller.
    if model_type in {"naive_bayes", "nb"}:
        classifier = MultinomialNB()
    elif model_type in {"logistic_regression", "lr", "logreg"}:
        classifier = LogisticRegression(max_iter=2000)
    else:
        raise ValueError("model_type must be 'naive_bayes' or 'logistic_regression'")

    # Step 3: Chain vectorizer and classifier into one reusable pipeline.
    return Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier),
    ])


def train_model(
    df: pd.DataFrame,
    model_type: str = "naive_bayes",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, pd.Series, pd.Series]:
    """Split data, train the selected model, and return model with test split."""
    # Step 1: Prepare features (email text) and labels (spam/ham).
    X = df["email"]
    y = df["label"]

    # Step 2: Create train/test split for unbiased evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Step 3: Build and train the machine learning pipeline.
    pipeline = _build_pipeline(model_type=model_type)
    pipeline.fit(X_train, y_train)
    return pipeline, X_test, y_test


def evaluate_model(model: Pipeline, X_test: pd.Series, y_test: pd.Series) -> Dict[str, object]:
    """Evaluate model on test data and return key metrics."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix,
    }


def predict_email(model: Pipeline, email_text: str) -> str:
    """Predict whether a single email is spam or ham."""
    prediction = model.predict([email_text])[0]
    return str(prediction)
