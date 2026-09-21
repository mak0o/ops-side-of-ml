"""推論ログとベースラインを比較してドリフトを検知する。

ベースラインの取得元は2つ。
- ローカル: MLflow の production alias が指す run の artifact
- AWS: model.tar.gz に同梱した baseline.json（--model-artifact で指定）
"""

import argparse
import io
import json
import os
import tarfile
from urllib.parse import urlparse

import boto3
import numpy as np
import pandas as pd

from src.baseline import interpret, psi

BUCKET = os.getenv("INFERENCE_LOG_BUCKET", "mlflow-bucket")
PREFIX = os.getenv("INFERENCE_LOG_PREFIX", "inference-logs")
MODEL_NAME = "cost-anomaly-detector"


def load_baseline_from_mlflow(alias: str) -> dict:
    """production alias が指すモデルの run から baseline.json を取得する。"""
    # mlflow はローカル専用。AWS 側の実行環境に含めないため関数内で import する。
    import mlflow

    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, alias)
    path = mlflow.artifacts.download_artifacts(
        run_id=mv.run_id, artifact_path="baseline.json"
    )
    with open(path) as f:
        return json.load(f)


def load_baseline_from_artifact(artifact_uri: str) -> dict:
    """model.tar.gz に同梱した baseline.json を取り出す。

    学習時に run_sagemaker() がモデルと同じ tar に入れているので、
    モデルとベースラインの対応が崩れない。
    """
    parsed = urlparse(artifact_uri)
    s3 = boto3.client("s3")
    body = s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"].read()

    with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tar:
        member = tar.extractfile("baseline.json")
        if member is None:
            raise SystemExit(f"{artifact_uri} に baseline.json がありません")
        return json.load(member)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """推論ログを特徴量が最上位に並ぶ形に揃える。

    ローカルの inference_logger はネストした features を書くが、
    Batch Transform の出力は特徴量が最上位に来る。
    混在してもよいよう、ファイル単位で揃えてから結合する。
    """
    if "features" in df.columns:
        features = pd.json_normalize(df["features"])
        return pd.concat([df.drop(columns=["features"]), features], axis=1)
    return df


def load_inference_logs(prefix: str) -> pd.DataFrame:
    s3 = boto3.client("s3", endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL") or None)
    paginator = s3.get_paginator("list_objects_v2")

    frames = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            body = s3.get_object(Bucket=BUCKET, Key=obj["Key"])["Body"].read()
            frames.append(_normalize(pd.read_json(io.BytesIO(body), lines=True)))

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias", default="production")
    parser.add_argument("--model-artifact", default=None,
                        help="指定すると MLflow ではなく model.tar.gz からベースラインを読む")
    parser.add_argument("--prefix", default=PREFIX,
                        help="推論ログのS3プレフィックス。日付で絞る場合に指定")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--min-samples", type=int, default=100,
                        help="これ未満は判定不能として終了コード 2 を返す")
    args = parser.parse_args()

    if args.model_artifact:
        baseline = load_baseline_from_artifact(args.model_artifact)
    else:
        baseline = load_baseline_from_mlflow(args.alias)

    logs = load_inference_logs(args.prefix)

    if logs.empty:
        print("no inference logs found")
        raise SystemExit(2)

    # PSI は10ビンの構成比で比べるので、件数が少ないと空のビンができて値が跳ねる。
    # 51件で同一母集団でも significant が出たため、判定不能として止める。
    if len(logs) < args.min_samples:
        print(f"samples {len(logs)} < {args.min_samples}: 判定に必要な件数に達していません")
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
    # 未捕捉の例外は終了コード 1 になり、「ドリフトあり」と区別できない。
    # クラッシュで再学習が走らないよう、想定外の失敗は判定不能 (2) にする。
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"drift_check failed: {type(e).__name__}: {e}")
        raise SystemExit(2) from e