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

# ① 登録と昇格判定に必須の指標。欠けていたら登録しない。
REQUIRED_METRICS = ["f2", "precision", "recall"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/cost.parquet"))
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--since", type=str, default="")
    parser.add_argument("--instance-type", type=str, default="ml.m5.large")
    parser.add_argument("--image-tag", type=str, default="latest")
    parser.add_argument("--wait", action="store_true", help="完了まで待つ")
    # ② 登録用の引数
    parser.add_argument("--register", action="store_true",
                        help="完了後に Model Package として登録する（--wait が必要）")
    parser.add_argument("--serve-image-tag", type=str, default=None,
                        help="登録するパッケージに紐づける推論イメージのタグ")
    args = parser.parse_args()

    # ③-1 引数の組み合わせを先に検査する。学習を始めてから気づくと課金が無駄になる。
    if args.register and not args.wait:
        raise SystemExit("--register には --wait が必要です")
    if args.register and not args.serve_image_tag:
        raise SystemExit("--register には --serve-image-tag が必要です")

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

    # ③-2 待たないならここで終わり
    if not args.wait:
        return

    desc = _wait(sm, job_name)

    # ③-3 完了したら登録する
    if args.register and desc["TrainingJobStatus"] == "Completed":
        serve_image = (
            f"{account}.dkr.ecr.{REGION}.amazonaws.com/{PROJECT}/serve:{args.serve_image_tag}"
        )
        _register(sm, desc, serve_image)


def _wait(sm, job_name: str) -> dict:
    """完了まで待ち、最後の describe 結果を返す。"""
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

            # ④ describe の結果を呼び出し元に返す（以前は return だけだった）
            return desc

        print(f"{status} ...")
        time.sleep(20)


# ⑤ ここに新しい関数を追加する。_wait() の後ろ、if __name__ の前。
def _register(sm, desc: dict, serve_image: str) -> str:
    """学習ジョブの成果物を Model Package として登録する。

    状態は PendingManualApproval。昇格するかは promote.py が決める。
    """
    metrics = {m["MetricName"]: m["Value"] for m in desc.get("FinalMetricDataList", [])}

    missing = [k for k in REQUIRED_METRICS if k not in metrics]
    if missing:
        print(f"ERROR: 指標が記録されていません: {', '.join(missing)}")
        print("登録しません。MetricDefinitions と学習ログを確認してください。")
        raise SystemExit(2)

    job_name = desc["TrainingJobName"]
    artifact = desc["ModelArtifacts"]["S3ModelArtifacts"]

    resp = sm.create_model_package(
        ModelPackageGroupName=MODEL_NAME,
        ModelPackageDescription=f"training job: {job_name}",
        InferenceSpecification={
            "Containers": [{"Image": serve_image, "ModelDataUrl": artifact}],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
        },
        ModelApprovalStatus="PendingManualApproval",
        # 値は文字列しか持てない。promote.py 側で float に戻す。
        CustomerMetadataProperties={
            **{k: f"{v:.4f}" for k, v in metrics.items()},
            "training_job": job_name,
        },
    )

    arn = resp["ModelPackageArn"]
    print(f"registered: {arn} (PendingManualApproval)")
    return arn


if __name__ == "__main__":
    main()