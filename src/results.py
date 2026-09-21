# src/results.py
"""判定結果を S3 に JSON で書く。

Processing Job は終了コード 0 以外を全て「ジョブ失敗」として扱う。
「ドリフトあり」「昇格拒否」のような正常な判定結果は、終了コードではなくファイルで返す。
"""

import json
from urllib.parse import urlparse

import boto3


def write_json(uri: str, payload: dict) -> None:
    parsed = urlparse(uri)
    boto3.client("s3").put_object(
        Bucket=parsed.netloc,
        Key=parsed.path.lstrip("/"),
        Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )