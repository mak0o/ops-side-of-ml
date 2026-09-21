"""ホールドアウト（学習に使っていない直近期間のラベル付きデータ）での評価。

昇格判定では、現行モデルと候補モデルをこのデータで推論し、同じ関数で指標を計算して比べる。
登録時の指標は各モデルが別々のデータで測ったものなので、比較には使わない。
"""

import io
from urllib.parse import urlparse

import boto3
import pandas as pd

from src.metrics import compute_metrics

TARGET = "is_anomaly"

# F2 は正例の件数に強く依存する。少なすぎると数件の差で結果が振れる。
MIN_POSITIVES = 10


def load_holdout(prefix_uri: str, s3=None) -> pd.DataFrame:
    """S3 プレフィクス配下の JSON Lines を全て読んで結合する。"""
    s3 = s3 or boto3.client("s3")
    parsed = urlparse(prefix_uri)
    bucket, prefix = parsed.netloc, parsed.path.lstrip("/")

    frames = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
            frames.append(pd.read_json(io.BytesIO(body), lines=True))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def check_holdout(df: pd.DataFrame, min_positives: int = MIN_POSITIVES) -> None:
    """判定に使えるデータか確かめる。使えなければ判定不能 (2) で止める。"""
    if df.empty:
        print("ERROR: ホールドアウトが空です")
        raise SystemExit(2)
    if TARGET not in df.columns:
        print(f"ERROR: ホールドアウトに正解ラベル '{TARGET}' がありません")
        raise SystemExit(2)

    positives = int(df[TARGET].sum())
    print(f"holdout: {len(df)} rows, {positives} anomalies")
    if positives < min_positives:
        print(f"ERROR: 異常の件数 {positives} < {min_positives}: 判定に必要な件数に達していません")
        raise SystemExit(2)


def score(model, features: list[str], df: pd.DataFrame) -> dict:
    """モデルをホールドアウトで推論し、指標を返す。列順は学習時の features.json に従う。"""
    X = df[features]
    y = df[TARGET]
    return compute_metrics(y, model.predict(X), model.predict_proba(X)[:, 1])