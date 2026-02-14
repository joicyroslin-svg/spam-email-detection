from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure console logging once for CLI scripts."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def resolve_data_path(path: Path | None = None) -> Path:
    """Resolve dataset path with backward-compatible fallbacks."""
    candidates = [path] if path else [Path("data/sample_emails.csv"), Path("dataset/sample_emails.csv")]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    checked = ", ".join(str(c) for c in candidates if c)
    raise FileNotFoundError(f"Dataset file not found. Checked: {checked}")


def resolve_model_path(path: Path | None = None) -> Path:
    """Resolve model path with backward-compatible fallbacks."""
    if path:
        return path
    if Path("models/spam_model.joblib").exists():
        return Path("models/spam_model.joblib")
    return Path("models/spam_model.joblib")


def load_data(file_path: Path | str) -> pd.DataFrame:
    """Load CSV data and validate required columns."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = pd.read_csv(path)
    required = {"email", "label"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    cleaned = data[["email", "label"]].dropna()
    if cleaned.empty:
        raise ValueError("Dataset is empty after removing missing values.")

    logger.info("Loaded dataset from %s with %d rows", path, len(cleaned))
    return cleaned


def preprocess_text(text: str) -> str:
    """Normalize text for vectorization."""
    normalized = str(text).lower()
    normalized = re.sub(r"https?://\\S+|www\\.\\S+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\\s]", " ", normalized)
    normalized = re.sub(r"\\s+", " ", normalized).strip()
    return normalized


def build_pipeline(model_type: str) -> Pipeline:
    """Create a sklearn pipeline with TF-IDF + selected classifier."""
    if model_type == "naive_bayes":
        classifier = MultinomialNB()
    elif model_type == "logistic_regression":
        classifier = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    else:
        raise ValueError("model_type must be 'naive_bayes' or 'logistic_regression'")

    return Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    preprocessor=preprocess_text,
                    ngram_range=(1, 2),
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def train_model(
    data: pd.DataFrame,
    model_type: str = "logistic_regression",
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[Pipeline, pd.Series, pd.Series, Dict[str, Any]]:
    """Train a single model and return fitted model, test split, and metrics."""
    X = data["email"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = build_pipeline(model_type)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", pos_label="spam")
    metrics = {
        "model_type": model_type,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    logger.info("Trained %s with F1=%.4f", model_type, metrics["f1"])
    return model, X_test, y_test, metrics


def compare_models(
    data: pd.DataFrame,
    candidates: Tuple[str, ...] = ("naive_bayes", "logistic_regression"),
) -> Tuple[Pipeline, str, pd.Series, pd.Series, Dict[str, Any]]:
    """Train candidate models and return the best by F1 score."""
    best_model: Pipeline | None = None
    best_name = ""
    best_metrics: Dict[str, Any] = {}
    best_X_test = pd.Series(dtype=str)
    best_y_test = pd.Series(dtype=str)

    for model_name in candidates:
        model, X_test, y_test, metrics = train_model(data, model_type=model_name)
        logger.info("Model %s comparison F1=%.4f", model_name, metrics["f1"])
        if not best_metrics or metrics["f1"] > best_metrics["f1"]:
            best_model = model
            best_name = model_name
            best_metrics = metrics
            best_X_test = X_test
            best_y_test = y_test

    if best_model is None:
        raise RuntimeError("Model comparison failed to produce a model.")

    logger.info("Best model selected: %s (F1=%.4f)", best_name, best_metrics["f1"])
    return best_model, best_name, best_X_test, best_y_test, best_metrics


def save_model(model: Pipeline, model_path: Path | str) -> Path:
    """Persist trained model to disk."""
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)
    return path


def load_model(model_path: Path | str) -> Pipeline:
    """Load a trained model from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    logger.info("Loaded model from %s", path)
    return model


def predict_email(model: Pipeline, email_text: str) -> Dict[str, Any]:
    """Predict email label and confidence when available."""
    prediction = str(model.predict([email_text])[0])

    confidence = None
    clf = model.named_steps.get("classifier")
    if hasattr(clf, "predict_proba"):
        classes = list(clf.classes_)
        spam_idx = classes.index("spam") if "spam" in classes else 1
        confidence = float(model.predict_proba([email_text])[0][spam_idx])
    elif hasattr(model, "decision_function"):
        margin = float(model.decision_function([email_text])[0])
        confidence = float(1.0 / (1.0 + pow(2.718281828, -margin)))

    return {"label": prediction, "confidence": confidence}


def evaluate_model(
    model: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
    reports_dir: Path | str = Path("reports"),
) -> Dict[str, Any]:
    """Evaluate model, save text report and confusion matrix image."""
    y_pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", pos_label="spam")
    cm = confusion_matrix(y_test, y_pred)
    report_text = classification_report(y_test, y_pred)

    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    evaluation_txt = reports_path / "evaluation.txt"
    evaluation_body = (
        f"Precision: {precision:.4f}\\n"
        f"Recall: {recall:.4f}\\n"
        f"F1-score: {f1:.4f}\\n\\n"
        f"Classification Report:\\n{report_text}\\n"
        f"Confusion Matrix:\\n{cm}\\n"
    )
    evaluation_txt.write_text(evaluation_body, encoding="utf-8")

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_image = reports_path / "confusion_matrix.png"
    plt.savefig(cm_image)
    plt.close()

    logger.info("Saved evaluation report to %s", evaluation_txt)
    logger.info("Saved confusion matrix image to %s", cm_image)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "classification_report": report_text,
        "confusion_matrix": cm,
        "evaluation_path": evaluation_txt,
        "confusion_matrix_path": cm_image,
    }
