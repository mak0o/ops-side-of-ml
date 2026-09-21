"""推論ログとベースラインを比較してドリフトを検知する。"""

import argparse
import io
import json
import os

import boto3
import mlflow
import numpy as np
import pandas as pd

from src.baseline import interpret, psi

BUCKET = os.getenv("INFERENCE_LOG_BUCKET", "mlflow-bucket")
PREFIX = os.getenv("INFERENCE_LOG_PREFIX", "inference-logs")
MODEL_NAME = "cost-anomaly-detector"


def load_baseline(alias: str) -> dict:
    """production alias が指すモデルの run から baseline.json を取得する。"""
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, alias)
    path = mlflow.artifacts.download_artifacts(
        run_id=mv.run_id, artifact_path="baseline.json"
    )
    with open(path) as f:
        return json.load(f)


def load_inference_logs(prefix: str) -> pd.DataFrame:
    s3 = boto3.client("s3", endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL"))
    paginator = s3.get_paginator("list_objects_v2")

    frames = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            body = s3.get_object(Bucket=BUCKET, Key=obj["Key"])["Body"].read()
            frames.append(pd.read_json(io.BytesIO(body), lines=True))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # ローカルの inference_logger はネストした features を書くが、
    # Batch Transform の出力は特徴量が最上位に来る。
    if "features" in df.columns:
        features = pd.json_normalize(df["features"])
        return pd.concat([df.drop(columns=["features"]), features], axis=1)

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias", default="production")
    parser.add_argument("--prefix", default=PREFIX,
                        help="推論ログのS3プレフィックス。日付で絞る場合に指定")
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()

    baseline = load_baseline(args.alias)
    logs = load_inference_logs(args.prefix)

    if logs.empty:
        print("no inference logs found")
        raise SystemExit(2)

    print(f"baseline: {baseline['n_samples']} samples")
    print(f"current:  {len(logs)} samples")
    print()
    print(f"{'feature':<20} {'PSI':>8}  {'status':<12} {'base_mean':>10} {'curr_mean':>10}")
    print("-" * 66)

    drifted = []
    for col, stats in baseline["features"].items():
        if col not in logs.columns:
            continue
        values = logs[col].to_numpy(dtype=float)
        score = psi(stats, values)
        status = interpret(score)
        if score >= args.threshold:
            drifted.append(col)
        print(
            f"{col:<20} {score:>8.4f}  {status:<12} "
            f"{stats['mean']:>10.2f} {np.mean(values):>10.2f}"
        )

    print()
    if drifted:
        print(f"DRIFT DETECTED: {', '.join(drifted)}")
        raise SystemExit(1)
    print("no significant drift")


if __name__ == "__main__":
    main()