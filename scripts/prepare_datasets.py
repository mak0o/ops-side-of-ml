"""時系列データを期間で分割し、パイプラインの入力として S3 に置く。

ワークフローの外にある「データを置く役」。本番なら Cost and Usage Report の
エクスポートとラベル付けの仕組みが担う位置。

  最後の N 日 ─┬─ 特徴量のみ     → batch-input/<name>/input.jsonl   （推論に流す）
              └─ 特徴量 + ラベル → eval-input/<name>/holdout.jsonl （ホールドアウト）
  それより前 ─────────────────→ training-input/<name>/train.parquet（再学習）

ホールドアウトは期間で切る。ランダムに切ると、移動平均の特徴量を通じて
隣接日の情報が学習側に漏れる。
"""

import argparse
import io
from pathlib import Path

import boto3
import pandas as pd

REGION = "ap-northeast-1"
PROJECT = "ops-side-of-ml"

FEATURES = [
    "cost",
    "cost_ma7",
    "cost_std7",
    "cost_ratio_ma7",
    "cost_vs_lastweek",
    "day_of_week",
    "is_weekend",
]
TARGET = "is_anomaly"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--name", required=True, help="S3 上のデータセット名")
    parser.add_argument("--holdout-days", type=int, default=60)
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    cutoff = df["date"].max() - pd.Timedelta(days=args.holdout_days - 1)
    train = df[df["date"] < cutoff]
    recent = df[df["date"] >= cutoff]

    print(f"train:   {train['date'].min():%Y-%m-%d} .. {train['date'].max():%Y-%m-%d}"
          f"  {len(train)} rows, {int(train[TARGET].sum())} anomalies")
    print(f"holdout: {recent['date'].min():%Y-%m-%d} .. {recent['date'].max():%Y-%m-%d}"
          f"  {len(recent)} rows, {int(recent[TARGET].sum())} anomalies")

    account = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]
    bucket = f"{PROJECT}-{account}"
    s3 = boto3.client("s3", region_name=REGION)

    buf = io.BytesIO()
    train.to_parquet(buf, index=False)
    uploads = {
        f"training-input/{args.name}/train.parquet": buf.getvalue(),
        f"batch-input/{args.name}/input.jsonl":
            recent[FEATURES].to_json(orient="records", lines=True).encode("utf-8"),
        f"eval-input/{args.name}/holdout.jsonl":
            recent[FEATURES + [TARGET]].to_json(orient="records", lines=True).encode("utf-8"),
    }

    print()
    for key, body in uploads.items():
        s3.put_object(Bucket=bucket, Key=key, Body=body)
        print(f"uploaded: s3://{bucket}/{key}")


if __name__ == "__main__":
    main()