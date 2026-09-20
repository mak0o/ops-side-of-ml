"""推論 API。

ローカル: MLflow Model Registry の @production からモデルを読む。
SageMaker: /opt/ml/model に展開された model.joblib を読む。

SageMaker が要求する /ping と /invocations を提供しつつ、
ローカルで使っている /health と /predict も残している。
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference_logger import InferenceLogger

MODEL_URI = os.getenv("MODEL_URI", "models:/cost-anomaly-detector@production")

# SageMaker は model.tar.gz をここに展開する
SM_MODEL = Path("/opt/ml/model")

logger = InferenceLogger()
model = None
model_version: str | None = None
model_error: str | None = None


def in_sagemaker() -> bool:
    return (SM_MODEL / "model.joblib").exists()


def _load_from_sagemaker() -> tuple[object, str]:
    import joblib

    clf = joblib.load(SM_MODEL / "model.joblib")

    # モデルと同じ tar に入っている metrics.json は学習時の情報なので
    # バージョン識別には使えない。SageMaker 側が渡す環境変数を使う。
    version = os.getenv("MODEL_PACKAGE_VERSION", "unknown")
    return clf, version


def _load_from_mlflow() -> tuple[object, str]:
    import mlflow

    clf = mlflow.sklearn.load_model(MODEL_URI)
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias("cost-anomaly-detector", "production")
    return clf, mv.version


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_version, model_error

    try:
        if in_sagemaker():
            model, model_version = _load_from_sagemaker()
        else:
            model, model_version = _load_from_mlflow()
        print(f"loaded: version {model_version}")
    except Exception as e:
        model_error = str(e)
        print(f"model load failed: {e}")

    yield

    logger.flush()


app = FastAPI(title="cost-anomaly-detector", lifespan=lifespan)


class CostRecord(BaseModel):
    cost: float
    cost_ma7: float
    cost_std7: float
    cost_ratio_ma7: float
    cost_vs_lastweek: float
    day_of_week: int = Field(ge=0, le=6)
    is_weekend: int = Field(ge=0, le=1)


def _predict(record: CostRecord) -> dict:
    if model is None:
        raise HTTPException(503, f"model not loaded: {model_error}")

    features = record.model_dump()
    df = pd.DataFrame([features])
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    logger.log(features, prediction, probability, model_version)

    return {"is_anomaly": prediction, "probability": probability}


# --- ローカル用 ---

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if model is not None else "degraded",
        "model_uri": MODEL_URI if not in_sagemaker() else str(SM_MODEL),
        "model_version": model_version,
    }


@app.post("/predict")
def predict(record: CostRecord) -> dict:
    return _predict(record)


# --- SageMaker 用 ---
# パス名は SageMaker の規約で固定されている。

@app.get("/ping")
def ping() -> dict:
    if model is None:
        raise HTTPException(503, "model not loaded")
    return {"status": "ok"}


@app.post("/invocations")
def invocations(record: CostRecord) -> dict:
    return _predict(record)