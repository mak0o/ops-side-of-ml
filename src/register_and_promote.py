"""学習ジョブの成果物を Model Package として登録し、そのまま昇格判定する。

Step Functions から Processing Job として呼ぶ。登録と判定を別ジョブにすると、
登録で得た ARN を次に渡すために S3 経由の受け渡しが1段増えるので、1つにまとめている。

--eval-s3-uri を渡すと、現行モデルと候補モデルを同じホールドアウトで評価して比べる。
渡さなければ登録時の指標で比べる（各モデルが別々のデータで測った値なので、比較としては弱い）。

終了コード
  0: 判定まで完了。Approved / Rejected のどちらも正常な結果で、内容は --result-s3-uri に書く
  2: 判定できなかった（学習が未完了、指標の欠落、ホールドアウトの不足、API の失敗など）
"""

import argparse
import json

import boto3

from src.holdout import MIN_POSITIVES, check_holdout, load_holdout, score
from src.promote import MIN_PRECISION, MIN_RECALL, decide_package, promote_package
from src.registry import (
    MODEL_PACKAGE_GROUP,
    REGION,
    container,
    latest_approved,
    load_model,
    register_training_job,
)
from src.results import write_json


def _rounded(metrics: dict | None) -> dict | None:
    if metrics is None:
        return None
    return {k: round(v, 4) for k, v in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-job", required=True)
    parser.add_argument("--serve-image", required=True,
                        help="パッケージに紐づける推論イメージの URI（タグ付き）")
    parser.add_argument("--group", default=MODEL_PACKAGE_GROUP)
    parser.add_argument("--eval-s3-uri", default=None,
                        help="ホールドアウト（ラベル付き JSON Lines）の S3 プレフィクス")
    parser.add_argument("--min-positives", type=int, default=MIN_POSITIVES)
    parser.add_argument("--result-s3-uri", default=None)
    parser.add_argument("--min-precision", type=float, default=MIN_PRECISION)
    parser.add_argument("--min-recall", type=float, default=MIN_RECALL)
    args = parser.parse_args()

    sm = boto3.client("sagemaker", region_name=REGION)

    desc = sm.describe_training_job(TrainingJobName=args.training_job)
    if desc["TrainingJobStatus"] != "Completed":
        print(f"学習ジョブが完了していません: {desc['TrainingJobStatus']}")
        raise SystemExit(2)

    if not args.eval_s3_uri:
        arn = register_training_job(sm, desc, args.serve_image, args.group)
        print()
        approved = promote_package(sm, arn, args.min_precision, args.min_recall)
        result = {"model_package_arn": arn, "approved": approved, "basis": "registered"}
        _finish(result, args.result_s3_uri)
        return

    # ホールドアウトは登録より先に検査する。使えないと分かっているのに登録すると、
    # 判定されないまま PendingManualApproval のパッケージが残る。
    holdout = load_holdout(args.eval_s3_uri)
    check_holdout(holdout, args.min_positives)

    arn = register_training_job(sm, desc, args.serve_image, args.group)
    cand_desc = sm.describe_model_package(ModelPackageName=arn)
    cand_label = f"v{cand_desc['ModelPackageVersion']}"

    cand_model, cand_features = load_model(desc["ModelArtifacts"]["S3ModelArtifacts"])
    cand = score(cand_model, cand_features, holdout)

    cur_desc = latest_approved(sm, args.group, exclude_arn=arn)
    if cur_desc is None:
        curr, curr_label = None, None
    else:
        cur_model, cur_features = load_model(container(cur_desc)["ModelDataUrl"])
        curr = score(cur_model, cur_features, holdout)
        curr_label = f"v{cur_desc['ModelPackageVersion']}"

    # ホールドアウトでの成績を候補のパッケージにも残す。登録時の指標とは別の名前で持つ。
    sm.update_model_package(
        ModelPackageArn=arn,
        CustomerMetadataProperties={f"holdout_{k}": f"{v:.4f}" for k, v in cand.items()},
    )

    print()
    approved = decide_package(sm, arn, curr, curr_label, cand, cand_label,
                              args.min_precision, args.min_recall, basis="holdout")

    # 候補の判定とは別に、現行モデルが今のデータで下限を割っていないかを記録する。
    # 候補が拒否され、かつ現行も劣化している状態を黙って放置しないため。
    current_degraded = curr is not None and (
        curr["precision"] < args.min_precision or curr["recall"] < args.min_recall
    )
    if current_degraded:
        print(f"WARNING: 現行モデル {curr_label} がホールドアウトで下限を割っています")

    result = {
        "model_package_arn": arn,
        "approved": approved,
        "basis": "holdout",
        "holdout_rows": len(holdout),
        "holdout_anomalies": int(holdout["is_anomaly"].sum()),
        "candidate_holdout": _rounded(cand),
        "current": curr_label,
        "current_holdout": _rounded(curr),
        "current_degraded": current_degraded,
    }
    _finish(result, args.result_s3_uri)


def _finish(result: dict, result_s3_uri: str | None) -> None:
    print()
    print(json.dumps(result, ensure_ascii=False))
    if result_s3_uri:
        write_json(result_s3_uri, result)
        print(f"result: {result_s3_uri}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"register_and_promote failed: {type(e).__name__}: {e}")
        raise SystemExit(2) from e