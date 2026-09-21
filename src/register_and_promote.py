# src/register_and_promote.py
"""学習ジョブの成果物を Model Package として登録し、そのまま昇格判定する。

Step Functions から Processing Job として呼ぶ。登録と判定を別ジョブにすると、
登録で得た ARN を次に渡すために S3 経由の受け渡しが1段増えるので、1つにまとめている。

終了コード
  0: 判定まで完了。Approved / Rejected のどちらも正常な結果で、内容は --result-s3-uri に書く
  2: 判定できなかった（学習が未完了、指標の欠落、API の失敗など）
"""

import argparse
import json

import boto3

from src.promote import MIN_PRECISION, MIN_RECALL, promote_package
from src.registry import MODEL_PACKAGE_GROUP, REGION, register_training_job
from src.results import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-job", required=True)
    parser.add_argument("--serve-image", required=True,
                        help="パッケージに紐づける推論イメージの URI（タグ付き）")
    parser.add_argument("--group", default=MODEL_PACKAGE_GROUP)
    parser.add_argument("--result-s3-uri", default=None)
    parser.add_argument("--min-precision", type=float, default=MIN_PRECISION)
    parser.add_argument("--min-recall", type=float, default=MIN_RECALL)
    args = parser.parse_args()

    sm = boto3.client("sagemaker", region_name=REGION)

    desc = sm.describe_training_job(TrainingJobName=args.training_job)
    if desc["TrainingJobStatus"] != "Completed":
        print(f"学習ジョブが完了していません: {desc['TrainingJobStatus']}")
        raise SystemExit(2)

    arn = register_training_job(sm, desc, args.serve_image, args.group)
    print()
    approved = promote_package(sm, arn, args.min_precision, args.min_recall)

    result = {"model_package_arn": arn, "approved": approved}
    print()
    print(json.dumps(result))
    if args.result_s3_uri:
        write_json(args.result_s3_uri, result)
        print(f"result: {args.result_s3_uri}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"register_and_promote failed: {type(e).__name__}: {e}")
        raise SystemExit(2) from e