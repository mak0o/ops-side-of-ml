"""SageMaker Training Job を起動して学習を実行する。

Training Job は一度実行して終了するリソースなので Terraform では管理せず、
このスクリプトから boto3 で起動する。
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import boto3

REGION = "ap-northeast-1"
PROJECT = "ops-side-of-ml"
MODEL_NAME = "cost-anomaly-detector"

# SageMaker が標準出力からメトリクスを拾うための正規表現。
# src/train.py の run_sagemaker() が出す key=value; 形式に対応する。
METRIC_DEFINITIONS = [
    # average_precision=... の行が precision としても拾われないよう行頭で固定する
    {"Name": name, "Regex": rf"^{name}=([0-9\.]+);"}
    for name in ["precision", "recall", "f1", "f2", "average_precision"]
]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/cost.parquet"))
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--since", type=str, default="")
    parser.add_argument("--instance-type", type=str, default="ml.m5.large")
    parser.add_argument("--image-tag", type=str, default="latest")
    parser.add_argument("--wait", action="store_true", help="完了まで待つ")
    args = parser.parse_args()

    sts = boto3.client("sts", region_name=REGION)
    account = sts.get_caller_identity()["Account"]

    bucket = f"{PROJECT}-{account}"
    image = f"{account}.dkr.ecr.{REGION}.amazonaws.com/{PROJECT}/train:{args.image_tag}"
    role = f"arn:aws:iam::{account}:role/{PROJECT}-sagemaker-execution"

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    job_name = f"{MODEL_NAME}-{stamp}"

    # 学習データを S3 に置く。ジョブごとにプレフィクスを分けて、
    # どのデータで学習したかを後から追えるようにする。
    s3 = boto3.client("s3", region_name=REGION)
    data_key = f"training-input/{job_name}/{args.data.name}"
    s3.upload_file(str(args.data), bucket, data_key)
    print(f"uploaded: s3://{bucket}/{data_key}")

    sm = boto3.client("sagemaker", region_name=REGION)
    sm.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": image,
            "TrainingInputMode": "File",
            "MetricDefinitions": METRIC_DEFINITIONS,
        },
        RoleArn=role,
        InputDataConfig=[
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{bucket}/training-input/{job_name}/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
        ],
        OutputDataConfig={"S3OutputPath": f"s3://{bucket}/training-output/"},
        ResourceConfig={
            "InstanceType": args.instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 10,
        },
        # ハイパーパラメータは文字列でしか渡せない。
        # src/train.py 側で int に変換している。
        HyperParameters={
            "n_estimators": str(args.n_estimators),
            "max_depth": str(args.max_depth),
            "since": args.since,
        },
        # 暴走したジョブが課金を垂れ流さないよう上限を切る。
        StoppingCondition={"MaxRuntimeInSeconds": 1800},
        Tags=[
            {"Key": "Project", "Value": PROJECT},
            {"Key": "ManagedBy", "Value": "script"},
        ],
    )

    print(f"started: {job_name}")
    print(
        "console: https://"
        f"{REGION}.console.aws.amazon.com/sagemaker/home?region={REGION}"
        f"#/jobs/{job_name}"
    )

    if args.wait:
        _wait(sm, job_name)


def _wait(sm, job_name: str) -> None:
    while True:
        desc = sm.describe_training_job(TrainingJobName=job_name)
        status = desc["TrainingJobStatus"]

        if status in ("Completed", "Failed", "Stopped"):
            print(f"\n{status}")

            if status == "Completed":
                metrics = {
                    m["MetricName"]: m["Value"]
                    for m in desc.get("FinalMetricDataList", [])
                }
                print(json.dumps(metrics, indent=2))
                print(f"artifact: {desc['ModelArtifacts']['S3ModelArtifacts']}")
                seconds = desc.get("BillableTimeInSeconds", 0)
                print(f"billable: {seconds}s")
            else:
                print(desc.get("FailureReason", "(理由不明)"))

            return

        print(f"{status} ...")
        time.sleep(20)


if __name__ == "__main__":
    main()