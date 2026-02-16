from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.spam_pipeline import load_model, predict_email

app = FastAPI(title="Spam Email Detection API", version="1.0.0")
MODEL_PATH = Path("models/spam_model.joblib")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=3, description="Email text to classify")


class PredictResponse(BaseModel):
    label: str
    confidence: float | None
    top_terms: list[str]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="Model not found. Train the model first.")

    model = load_model(MODEL_PATH)
    result = predict_email(model, payload.text)
    terms = [term for term, _ in result.get("top_terms", [])]

    return PredictResponse(label=result["label"], confidence=result.get("confidence"), top_terms=terms)
