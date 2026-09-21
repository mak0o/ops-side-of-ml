# src/registry.py
"""SageMaker Model Registry の参照。

ローカルの MLflow alias (@production) に相当する「現行モデル」の解決をここにまとめる。
現行 = グループ内で作成日時が最新の Approved パッケージ。
"""

REGION = "ap-northeast-1"
MODEL_PACKAGE_GROUP = "cost-anomaly-detector"


def latest_approved(sm, group: str, exclude_arn: str | None = None) -> dict | None:
    """グループ内で最新の Approved パッケージの describe 結果を返す。無ければ None。

    exclude_arn は昇格判定で候補自身を比較対象から外すために使う。
    作成日時で並べるので、古いパッケージを後から Approved にしても現行にはならない。
    """
    resp = sm.list_model_packages(
        ModelPackageGroupName=group,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=10,
    )
    for summary in resp["ModelPackageSummaryList"]:
        if summary["ModelPackageArn"] == exclude_arn:
            continue
        return sm.describe_model_package(ModelPackageName=summary["ModelPackageArn"])
    return None


def container(desc: dict) -> dict:
    """パッケージの推論コンテナ定義（Image と ModelDataUrl）を返す。"""
    return desc["InferenceSpecification"]["Containers"][0]


# 登録と昇格判定に必須の指標。欠けていたら登録しない。
REQUIRED_METRICS = ["f2", "precision", "recall"]


def register_training_job(sm, desc: dict, serve_image: str,
                          group: str = MODEL_PACKAGE_GROUP) -> str:
    """学習ジョブの成果物を Model Package として登録し、ARN を返す。

    desc は describe_training_job の結果。状態は PendingManualApproval で、
    昇格するかは promote が決める。
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
        ModelPackageGroupName=group,
        ModelPackageDescription=f"training job: {job_name}",
        InferenceSpecification={
            "Containers": [{"Image": serve_image, "ModelDataUrl": artifact}],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
        },
        ModelApprovalStatus="PendingManualApproval",
        # 値は文字列しか持てない。promote 側で float に戻す。
        CustomerMetadataProperties={
            **{k: f"{v:.4f}" for k, v in metrics.items()},
            "training_job": job_name,
        },
    )

    arn = resp["ModelPackageArn"]
    print(f"registered: {arn} (PendingManualApproval)")
    return arn