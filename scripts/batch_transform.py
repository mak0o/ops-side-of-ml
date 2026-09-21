"""SageMaker Batch Transform でまとめて推論する。

Transform Job は一度実行して終了するリソースなので Terraform では管理せず、
このスクリプトから boto3 で起動する。Model も同様にここで作る。
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import boto3
import pandas as pd

REGION = "ap-northeast-1"
PROJECT = "ops-side-of-ml"
MODEL_NAME = "cost-anomaly-detector"

FEATURES = [
    "cost",
    "cost_ma7",
    "cost_std7",
    "cost_ratio_ma7",
    "cost_vs_lastweek",
    "day_of_week",
    "is_weekend",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/cost.parquet"))
    parser.add_argument("--limit", type=int, default=200,
                        help="推論する行数。SingleRecord なので 1 行 1 リクエストになる")
    parser.add_argument("--model-artifact", type=str, required=True)
    parser.add_argument("--image-tag", type=str, default="latest")
    parser.add_argument("--instance-type", type=str, default="ml.m5.large")
    parser.add_argument("--join-input", action="store_true",
                        help="出力に入力データを結合する（Step 2）")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()

    account = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]
    bucket = f"{PROJECT}-{account}"
    image = f"{account}.dkr.ecr.{REGION}.amazonaws.com/{PROJECT}/serve:{args.image_tag}"
    role = f"arn:aws:iam::{account}:role/{PROJECT}-sagemaker-execution"

    now = datetime.now(UTC)
    stamp = now.strftime("%Y%m%d-%H%M%S")
    job_name = f"{MODEL_NAME}-batch-{stamp}"

    # drift_check が日付プレフィクスで読めるよう、既存の推論ログと同じ構造にする。
    output_prefix = f"inference-logs/{now:year=%Y/month=%m/day=%d}/{job_name}"

    # --- 入力を JSON Lines に変換して S3 に置く ---
    # parquet のままでは Batch Transform が 1 レコードずつ切り出せない。
    df = pd.read_parquet(args.data).head(args.limit)[FEATURES]
    local = Path(f"/tmp/{job_name}.jsonl")
    local.write_text(
        "\n".join(json.dumps(r) for r in df.to_dict(orient="records"))
    )

    s3 = boto3.client("s3", region_name=REGION)
    input_key = f"batch-input/{job_name}/input.jsonl"
    s3.upload_file(str(local), bucket, input_key)
    print(f"uploaded: s3://{bucket}/{input_key} ({len(df)} rows)")

    sm = boto3.client("sagemaker", region_name=REGION)

    # --- Model を作る ---
    # エンドポイントを destroy したので Terraform 側には存在しない。
    # Transform Job と同じライフサイクルで扱う。
    model_name = f"{MODEL_NAME}-{stamp}"
    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        PrimaryContainer={
            "Image": image,
            "ModelDataUrl": args.model_artifact,
            "Environment": {
                # Batch Transform は出力を S3 に書くので自前のログ書き出しは不要。
                "INFERENCE_LOG_ENABLED": "0",
                "MODEL_VERSION": args.image_tag,
            },
        },
    )
    print(f"model: {model_name}")

    # --- Transform Job ---
    params = {
        "TransformJobName": job_name,
        "ModelName": model_name,
        "TransformInput": {
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"s3://{bucket}/batch-input/{job_name}/",
                }
            },
            # FastAPI の pydantic は application/json 以外を受け付けない。
            # SplitType=Line で 1 行ずつ切り出し、各行を JSON として送る。
            "ContentType": "application/json",
            "SplitType": "Line",
        },
        "TransformOutput": {
            "S3OutputPath": f"s3://{bucket}/{output_prefix}/",
            "Accept": "application/json",
            "AssembleWith": "Line",
        },
        "TransformResources": {
            "InstanceType": args.instance_type,
            "InstanceCount": 1,
        },
        "BatchStrategy": "SingleRecord",
        "Tags": [
            {"Key": "Project", "Value": PROJECT},
            {"Key": "ManagedBy", "Value": "script"},
        ],
    }

    if args.join_input:
        # 出力に入力データを結合する。予測結果は SageMakerOutput キーに入る。
        params["DataProcessing"] = {
            "InputFilter": "$",
            "JoinSource": "Input",
            "OutputFilter": "$",
        }

    sm.create_transform_job(**params)
    print(f"started: {job_name}")

    if not args.wait:
        print(f"note: --wait なしのため Model {model_name} は残る。完了後に削除すること")
        return

    _wait(sm, job_name, bucket)

    # Transform Job が終われば Model は不要。残すと溜まり続ける。
    sm.delete_model(ModelName=model_name)
    print(f"deleted model: {model_name}")

def _wait(sm, job_name: str, bucket: str) -> None:
    while True:
        desc = sm.describe_transform_job(TransformJobName=job_name)
        status = desc["TransformJobStatus"]

        if status in ("Completed", "Failed", "Stopped"):
            print(f"\n{status}")
            if status == "Completed":
                print(f"output: {desc['TransformOutput']['S3OutputPath']}")
                print(f"billable: {desc.get('BillableTimeInSeconds', 0)}s")
            else:
                print(desc.get("FailureReason", "(理由不明)"))
            return

        print(f"{status} ...")
        time.sleep(20)


if __name__ == "__main__":
    main()
