from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
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


def calibrate_pipeline(model: Pipeline, X_train: pd.Series, y_train: pd.Series, cv: int = 3) -> Pipeline:
    """Calibrate classifier probabilities with sigmoid calibration."""
    calibrated = CalibratedClassifierCV(estimator=model, cv=cv, method="sigmoid")
    calibrated.fit(X_train, y_train)
    logger.info("Applied probability calibration using CalibratedClassifierCV")
    return calibrated


def train_model(
    data: pd.DataFrame,
    model_type: str = "logistic_regression",
    test_size: float = 0.25,
    random_state: int = 42,
    calibrate: bool = False,
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
    if calibrate:
        model = calibrate_pipeline(model, X_train, y_train)
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", pos_label="spam")
    accuracy = accuracy_score(y_test, y_pred)
    metrics = {
        "model_type": model_type,
        "accuracy": float(accuracy),
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
    calibrate: bool = False,
) -> Tuple[Pipeline, str, pd.Series, pd.Series, Dict[str, Any]]:
    """Train candidate models and return the best by F1 score."""
    best_model: Pipeline | None = None
    best_name = ""
    best_metrics: Dict[str, Any] = {}
    best_X_test = pd.Series(dtype=str)
    best_y_test = pd.Series(dtype=str)

    for model_name in candidates:
        model, X_test, y_test, metrics = train_model(data, model_type=model_name, calibrate=calibrate)
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


def tune_hyperparameters_randomized(
    data: pd.DataFrame,
    model_type: str,
    cv_folds: int = 3,
    random_state: int = 42,
    n_iter: int = 10,
    reports_dir: Path | str = Path("reports"),
) -> Tuple[Pipeline, Dict[str, Any], float, Path]:
    """Tune model hyperparameters with RandomizedSearchCV and save tuning history."""
    X = data["email"]
    y = data["label"]

    base_model = build_pipeline(model_type)
    if model_type == "naive_bayes":
        param_distributions = {
            "vectorizer__min_df": [1, 2, 3],
            "vectorizer__ngram_range": [(1, 1), (1, 2)],
            "classifier__alpha": [0.01, 0.1, 0.5, 1.0, 2.0],
        }
    elif model_type == "logistic_regression":
        param_distributions = {
            "vectorizer__min_df": [1, 2, 3],
            "vectorizer__ngram_range": [(1, 1), (1, 2)],
            "classifier__C": [0.25, 0.5, 1.0, 2.0, 4.0],
        }
    else:
        raise ValueError("model_type must be 'naive_bayes' or 'logistic_regression'")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=make_scorer(f1_score, pos_label="spam"),
        cv=cv,
        random_state=random_state,
        n_jobs=None,
    )
    search.fit(X, y)

    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    tuning_path = reports_path / "tuning_results.csv"
    cv_df = pd.DataFrame(search.cv_results_)
    keep_cols = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "params",
    ]
    cv_df = cv_df[[c for c in keep_cols if c in cv_df.columns]].sort_values(by="rank_test_score")
    cv_df.to_csv(tuning_path, index=False)

    logger.info("Randomized tuning complete for %s. Best F1=%.4f", model_type, search.best_score_)
    logger.info("Best params: %s", search.best_params_)
    logger.info("Saved tuning history to %s", tuning_path)
    return search.best_estimator_, dict(search.best_params_), float(search.best_score_), tuning_path


def cross_validate_models(
    data: pd.DataFrame,
    candidates: Tuple[str, ...] = ("naive_bayes", "logistic_regression"),
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Run cross-validation for candidate models and return F1 summary stats."""
    X = data["email"]
    y = data["label"]
    scorer = make_scorer(f1_score, pos_label="spam")
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    summary: Dict[str, Dict[str, float]] = {}

    for model_name in candidates:
        model = build_pipeline(model_name)
        scores = cross_val_score(model, X, y, cv=splitter, scoring=scorer)
        summary[model_name] = {
            "cv_f1_mean": float(scores.mean()),
            "cv_f1_std": float(scores.std()),
        }
        logger.info(
            "CV %s | mean F1=%.4f | std=%.4f",
            model_name,
            summary[model_name]["cv_f1_mean"],
            summary[model_name]["cv_f1_std"],
        )

    return summary


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


def _get_base_pipeline(model: Pipeline) -> Pipeline:
    """Extract underlying Pipeline from calibrated or plain model."""
    if isinstance(model, Pipeline):
        return model
    if hasattr(model, "estimator") and isinstance(model.estimator, Pipeline):
        return model.estimator
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        candidate = getattr(model.calibrated_classifiers_[0], "estimator", None)
        if isinstance(candidate, Pipeline):
            return candidate
    raise ValueError("Unable to extract base pipeline for explainability.")


def get_top_global_terms(model: Pipeline, top_n: int = 20) -> List[Tuple[str, float]]:
    """Return global top spam terms from model coefficients/log-probabilities."""
    base = _get_base_pipeline(model)
    vec = base.named_steps["vectorizer"]
    clf = base.named_steps["classifier"]
    features = vec.get_feature_names_out()

    if hasattr(clf, "coef_"):
        scores = clf.coef_.ravel()
    elif hasattr(clf, "feature_log_prob_"):
        if len(clf.classes_) == 2 and "spam" in clf.classes_:
            spam_idx = list(clf.classes_).index("spam")
            ham_idx = 1 - spam_idx
            scores = clf.feature_log_prob_[spam_idx] - clf.feature_log_prob_[ham_idx]
        else:
            scores = clf.feature_log_prob_[0]
    else:
        return []

    top_idx = np.argsort(scores)[::-1][:top_n]
    return [(str(features[i]), float(scores[i])) for i in top_idx]


def get_top_terms_for_text(model: Pipeline, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
    """Return per-email influential terms for the spam class."""
    base = _get_base_pipeline(model)
    vec = base.named_steps["vectorizer"]
    clf = base.named_steps["classifier"]
    features = vec.get_feature_names_out()

    text_vec = vec.transform([text])
    if hasattr(clf, "coef_"):
        weights = clf.coef_.ravel()
    elif hasattr(clf, "feature_log_prob_"):
        classes = list(clf.classes_)
        spam_idx = classes.index("spam") if "spam" in classes else 1
        ham_idx = 1 - spam_idx
        weights = clf.feature_log_prob_[spam_idx] - clf.feature_log_prob_[ham_idx]
    else:
        return []

    contrib = text_vec.multiply(weights).toarray().ravel()
    idx = np.argsort(contrib)[::-1]
    results: List[Tuple[str, float]] = []
    for i in idx:
        if contrib[i] <= 0:
            continue
        results.append((str(features[i]), float(contrib[i])))
        if len(results) >= top_n:
            break
    return results


def save_explainability_report(
    model: Pipeline,
    sample_text: str,
    reports_dir: Path | str = Path("reports"),
    top_n: int = 20,
) -> Path:
    """Save global and sample-level explainability terms."""
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    global_terms = get_top_global_terms(model, top_n=top_n)
    sample_terms = get_top_terms_for_text(model, sample_text, top_n=min(10, top_n))

    lines = ["Global top spam terms:"]
    for term, score in global_terms:
        lines.append(f"- {term}: {score:.4f}")

    lines.append("\nSample text top terms:")
    if sample_terms:
        for term, score in sample_terms:
            lines.append(f"- {term}: {score:.4f}")
    else:
        lines.append("- No positive term contribution found")

    report_path = reports_path / "explainability_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved explainability report to %s", report_path)
    return report_path


def predict_email(model: Pipeline, email_text: str) -> Dict[str, Any]:
    """Predict email label and confidence when available."""
    prediction = str(model.predict([email_text])[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([email_text])[0]
        classes = list(model.classes_) if hasattr(model, "classes_") else ["ham", "spam"]
        spam_idx = classes.index("spam") if "spam" in classes else 1
        confidence = float(proba[spam_idx])
    elif hasattr(model, "decision_function"):
        margin = float(model.decision_function([email_text])[0])
        confidence = float(1.0 / (1.0 + np.exp(-margin)))

    top_terms = get_top_terms_for_text(model, email_text, top_n=8)
    return {"label": prediction, "confidence": confidence, "top_terms": top_terms}


def evaluate_model(
    model: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
    reports_dir: Path | str = Path("reports"),
) -> Dict[str, Any]:
    """Evaluate model, save text report and visualization artifacts."""
    y_pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", pos_label="spam")
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report_text = classification_report(y_test, y_pred)
    y_true_binary = (y_test == "spam").astype(int).to_numpy()

    # Use probability when available; otherwise convert decision function to pseudo-probability.
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        classes = list(model.classes_) if hasattr(model, "classes_") else ["ham", "spam"]
        spam_idx = classes.index("spam") if "spam" in classes else 1
        y_score = probs[:, spam_idx]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X_test)
        y_score = 1.0 / (1.0 + np.exp(-decision))
    else:
        y_score = y_true_binary.astype(float)

    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true_binary, y_score)
    f1_values = 2 * pr_precision[:-1] * pr_recall[:-1] / (pr_precision[:-1] + pr_recall[:-1] + 1e-12)
    best_idx = int(np.argmax(f1_values)) if len(f1_values) else 0
    best_threshold = float(pr_thresholds[best_idx]) if len(pr_thresholds) else 0.5
    roc_fpr, roc_tpr, _ = roc_curve(y_true_binary, y_score)
    roc_auc = float(auc(roc_fpr, roc_tpr))
    pr_auc = float(auc(pr_recall, pr_precision))

    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    evaluation_txt = reports_path / "evaluation.txt"
    evaluation_body = (
        f"Accuracy: {accuracy:.4f}\\n"
        f"Precision: {precision:.4f}\\n"
        f"Recall: {recall:.4f}\\n"
        f"F1-score: {f1:.4f}\\n\\n"
        f"Best threshold (by PR F1): {best_threshold:.4f}\\n"
        f"ROC-AUC: {roc_auc:.4f}\\n"
        f"PR-AUC: {pr_auc:.4f}\\n\\n"
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

    roc_image = reports_path / "roc_curve.png"
    plt.figure(figsize=(5, 4))
    plt.plot(roc_fpr, roc_tpr, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_image)
    plt.close()

    pr_image = reports_path / "pr_curve.png"
    plt.figure(figsize=(5, 4))
    plt.plot(pr_recall, pr_precision, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(pr_image)
    plt.close()

    logger.info("Saved evaluation report to %s", evaluation_txt)
    logger.info("Saved confusion matrix image to %s", cm_image)
    logger.info("Saved ROC curve image to %s", roc_image)
    logger.info("Saved PR curve image to %s", pr_image)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "classification_report": report_text,
        "confusion_matrix": cm,
        "best_threshold": best_threshold,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "evaluation_path": evaluation_txt,
        "confusion_matrix_path": cm_image,
        "roc_curve_path": roc_image,
        "pr_curve_path": pr_image,
    }


def evaluate_benchmark_datasets(
    model: Pipeline,
    dataset_paths: Iterable[Path | str],
    reports_dir: Path | str = Path("reports"),
) -> Path:
    """Evaluate model on multiple datasets and save benchmark table."""
    rows: List[Dict[str, Any]] = []
    for ds in dataset_paths:
        ds_path = Path(ds)
        if not ds_path.exists():
            continue
        data = load_data(ds_path)
        y_true = data["label"]
        y_pred = model.predict(data["email"])
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label="spam")
        accuracy = accuracy_score(y_true, y_pred)
        rows.append(
            {
                "dataset": str(ds_path),
                "rows": int(len(data)),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    out_csv = reports_path / "benchmark_results.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info("Saved benchmark results to %s", out_csv)
    return out_csv


def _safe_git_commit() -> str:
    """Get current git commit hash when available."""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return ""


def save_model_metadata(
    model: Pipeline,
    metrics: Dict[str, Any],
    data_path: Path | str,
    model_path: Path | str,
    reports_dir: Path | str = Path("reports"),
    best_params: Dict[str, Any] | None = None,
    extra_artifacts: Dict[str, str] | None = None,
) -> Path:
    """Save model metadata/version information to reports/model_metadata.json."""
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    data_file = Path(data_path)
    data_bytes = data_file.read_bytes() if data_file.exists() else b""
    dataset_sha256 = hashlib.sha256(data_bytes).hexdigest() if data_bytes else ""

    model_file = Path(model_path)
    model_sha256 = hashlib.sha256(model_file.read_bytes()).hexdigest() if model_file.exists() else ""

    classifier = None
    if hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier")
    elif hasattr(model, "estimator") and hasattr(model.estimator, "named_steps"):
        classifier = model.estimator.named_steps.get("classifier")

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _safe_git_commit(),
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "dataset_path": str(data_file),
        "dataset_sha256": dataset_sha256,
        "classifier": classifier.__class__.__name__ if classifier is not None else "unknown",
        "metrics": {
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "roc_auc": metrics.get("roc_auc"),
            "pr_auc": metrics.get("pr_auc"),
            "best_threshold": metrics.get("best_threshold"),
        },
        "best_params": best_params or {},
        "artifacts": extra_artifacts or {},
    }

    metadata_path = reports_path / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved model metadata to %s", metadata_path)
    return metadata_path
