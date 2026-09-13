"""Model Registry から production モデルをロードして推論する API。"""

import os

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference_logger import InferenceLogger

MODEL_URI = os.getenv("MODEL_URI", "models:/cost-anomaly-detector@production")

app = FastAPI(title="cost-anomaly-detector")

logger = InferenceLogger()
model_version: str | None = None

model = None
model_error: str | None = None


@app.on_event("startup")
def load_model() -> None:
    global model, model_error, model_version
    try:
        model = mlflow.sklearn.load_model(MODEL_URI)
        client = mlflow.MlflowClient()
        mv = client.get_model_version_by_alias("cost-anomaly-detector", "production")
        model_version = mv.version
        print(f"loaded: {MODEL_URI} (version {model_version})")
    except Exception as e:
        model_error = str(e)
        print(f"model load failed: {e}")


class CostRecord(BaseModel):
    cost: float
    cost_ma7: float
    cost_std7: float
    cost_ratio_ma7: float
    cost_vs_lastweek: float
    day_of_week: int = Field(ge=0, le=6)
    is_weekend: int = Field(ge=0, le=1)


@app.on_event("shutdown")
def flush_logs() -> None:
    logger.flush()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if model is not None else "degraded",
        "model_uri": MODEL_URI,
        "model_version": model_version,
    }


@app.post("/predict")
def predict(record: CostRecord) -> dict:
    if model is None:
        raise HTTPException(503, f"model not loaded: {model_error}")

    features = record.model_dump()
    df = pd.DataFrame([features])
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    logger.log(features, prediction, probability, model_version)

    return {"is_anomaly": prediction, "probability": probability}