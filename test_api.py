from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import MODEL_PATH, app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_success() -> None:
    if not MODEL_PATH.exists():
        pytest.skip("Model artifact not found for API integration test.")

    response = client.post("/predict", json={"text": "Win cash now! Click this link."})
    assert response.status_code == 200
    payload = response.json()
    assert "label" in payload
    assert "confidence" in payload
    assert "top_terms" in payload
    assert payload["label"] in {"spam", "ham"}


def test_predict_endpoint_model_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    missing_path = Path("models/nonexistent_model.joblib")
    monkeypatch.setattr("api.main.MODEL_PATH", missing_path)

    response = client.post("/predict", json={"text": "test message"})
    assert response.status_code == 404
    assert "Model not found" in response.json()["detail"]
