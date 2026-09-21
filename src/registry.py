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