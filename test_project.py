import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "spam_email_detection_project.py"
DATA = ROOT / "data" / "sample_emails.csv"
MODEL = ROOT / "models" / "spam_model.joblib"


def run_cmd(args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_train_creates_model():
    result = run_cmd(["train", "--data", str(DATA), "--model-out", str(MODEL)])
    assert "Saved model" in result.stdout
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
    assert "Prediction:" in result.stdout
    assert "Spam confidence:" in result.stdout


def test_analyze_outputs_risk_details():
    result = run_cmd(
        [
            "analyze",
            "--model",
            str(MODEL),
            "--text",
            "URGENT: Verify OTP now to avoid account suspension.",
        ]
    )
    assert "Rule-based risk score:" in result.stdout
    assert "Matched risk signals:" in result.stdout
