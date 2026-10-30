from pathlib import Path
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "iris_pipeline.joblib"

app = FastAPI(title="Real-Time ML Inference API", version="1.0.0",
              description="REST API for Iris flower classification.")

class PredictionRequest(BaseModel):
    sepal_length: float = Field(..., gt=0, le=10)
    sepal_width: float = Field(..., gt=0, le=10)
    petal_length: float = Field(..., gt=0, le=10)
    petal_width: float = Field(..., gt=0, le=10)

class PredictionResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]
    confidence: float
    model_version: str

if not MODEL_PATH.exists():
    raise RuntimeError("Model artifact missing. Run python train.py first.")
model = joblib.load(MODEL_PATH)
CLASSES = ["setosa", "versicolor", "virginica"]

@app.get("/")
def root():
    return {"message": "Real-Time ML Inference API", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "model_version": "1.0.0"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    try:
        values = np.array([[data.sepal_length, data.sepal_width,
                            data.petal_length, data.petal_width]])
        probs = model.predict_proba(values)[0]
        idx = int(np.argmax(probs))
        return PredictionResponse(
            prediction=CLASSES[idx],
            probabilities={CLASSES[i]: round(float(probs[i]), 6) for i in range(3)},
            confidence=round(float(probs[idx]), 6),
            model_version="1.0.0")
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc
