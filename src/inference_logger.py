"""推論ログを JSON Lines として S3 互換ストレージにバッファリング書き込みする。

SageMaker Data Capture に近い構造（日付パーティション + JSON Lines）にしておくことで、
AWS 移行時にドリフト検知のコードを大きく変えずに済ませる。
"""

import json
import os
import threading
import uuid
from datetime import UTC, datetime

import boto3

BUCKET = os.getenv("INFERENCE_LOG_BUCKET", "mlflow-bucket")
PREFIX = os.getenv("INFERENCE_LOG_PREFIX", "inference-logs")
FLUSH_SIZE = int(os.getenv("INFERENCE_LOG_FLUSH_SIZE", "10"))
# Batch Transform は出力を SageMaker が S3 に書くので、自前のログは不要。
ENABLED = os.getenv("INFERENCE_LOG_ENABLED", "1") == "1"


class InferenceLogger:
    def __init__(self) -> None:
        self._buffer: list[dict] = []
        self._lock = threading.Lock()
        self._s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL"),
        )

    def log(self, features: dict, prediction: int, probability: float,
            model_version: str | None) -> None:
        if not ENABLED:
            return     
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "model_version": model_version,
            "features": features,
            "prediction": prediction,
            "probability": probability,
        }
        with self._lock:
            self._buffer.append(record)
            if len(self._buffer) >= FLUSH_SIZE:
                pending, self._buffer = self._buffer, []
            else:
                pending = []

        if pending:
            self._upload(pending)

    def flush(self) -> None:
        with self._lock:
            pending, self._buffer = self._buffer, []
        if pending:
            self._upload(pending)

    def _upload(self, records: list[dict]) -> None:
        now = datetime.now(UTC)
        key = (
            f"{PREFIX}/year={now:%Y}/month={now:%m}/day={now:%d}/"
            f"{now:%H%M%S}-{uuid.uuid4().hex[:8]}.jsonl"
        )
        body = "\n".join(json.dumps(r) for r in records).encode("utf-8")
        try:
            self._s3.put_object(Bucket=BUCKET, Key=key, Body=body)
            print(f"flushed {len(records)} records -> s3://{BUCKET}/{key}")
        except Exception as e:
            # ログ書き込みの失敗で推論を止めない
            print(f"inference log upload failed: {e}")