import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "spam_email_detection_project.py"
DATA = ROOT / "data" / "sample_emails.csv"
MODEL = ROOT / "models" / "spam_model.joblib"
REPORT = ROOT / "reports" / "evaluation.txt"


def run_cmd(args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def combined_output(result):
    return (result.stdout or "") + (result.stderr or "")


def test_train_creates_model():
    result = run_cmd([
        "train",
        "--data",
        str(DATA),
        "--model-out",
        str(MODEL),
        "--model-type",
        "naive_bayes",
    ])
    output = combined_output(result)
    assert "Saved model" in output
    assert MODEL.exists()


def test_predict_outputs_label():
    result = run_cmd(
        [
            "predict",
            "--model",
            str(MODEL),
            "--text",
            "You won a free gift card, click now.",
        ]
    )
    output = combined_output(result)
    assert "Prediction:" in output


def test_evaluate_writes_report():
    run_cmd(["evaluate", "--model", str(MODEL), "--data", str(DATA)])
    assert REPORT.exists()


def test_train_with_cv_flag():
    result = run_cmd(
        [
            "train",
            "--data",
            str(DATA),
            "--model-out",
            str(MODEL),
            "--compare-models",
            "--cv-folds",
            "3",
        ]
    )
    output = combined_output(result)
    assert "Cross-validation summary" in output
