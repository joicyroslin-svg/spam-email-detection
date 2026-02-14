import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "spam_email_detection_project.py"
DATA = ROOT / "dataset" / "sample_emails.csv"
MODEL = ROOT / "model" / "spam_model.pkl"


def run_cmd(args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


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
    assert "Model saved to" in result.stdout
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
