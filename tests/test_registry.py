# tests/test_registry.py
"""Model Registry まわりのロジックのテスト。"""

import pytest

from src.registry import latest_approved, register_training_job


class FakeSageMaker:
    """list_model_packages の結果だけを返す偽物。approved は新しい順の ARN。"""

    def __init__(self, approved: list[str]):
        self.approved = approved

    def list_model_packages(self, **kwargs):
        # 並び順の指定が変わると「最新」の意味が壊れるので、ここで固定する。
        assert kwargs["ModelApprovalStatus"] == "Approved"
        assert kwargs["SortBy"] == "CreationTime"
        assert kwargs["SortOrder"] == "Descending"
        return {"ModelPackageSummaryList": [{"ModelPackageArn": a} for a in self.approved]}

    def describe_model_package(self, ModelPackageName):
        return {"ModelPackageArn": ModelPackageName}


class FakeRegistrar:
    """create_model_package の呼び出しを記録する偽物。"""

    def __init__(self):
        self.calls = []

    def create_model_package(self, **kwargs):
        self.calls.append(kwargs)
        return {"ModelPackageArn": "arn/new"}


def _training_desc(metrics: dict) -> dict:
    return {
        "TrainingJobName": "job-1",
        "ModelArtifacts": {"S3ModelArtifacts": "s3://bucket/model.tar.gz"},
        "FinalMetricDataList": [{"MetricName": k, "Value": v} for k, v in metrics.items()],
    }


def test_returns_newest_approved():
    sm = FakeSageMaker(["arn/3", "arn/1"])
    assert latest_approved(sm, "g")["ModelPackageArn"] == "arn/3"


def test_excludes_candidate_itself():
    """候補自身が Approved でも、比較対象からは外す。"""
    sm = FakeSageMaker(["arn/3", "arn/1"])
    assert latest_approved(sm, "g", exclude_arn="arn/3")["ModelPackageArn"] == "arn/1"


def test_none_when_no_approved():
    assert latest_approved(FakeSageMaker([]), "g") is None


def test_register_stores_metrics_as_strings():
    sm = FakeRegistrar()
    desc = _training_desc({"f2": 0.815, "precision": 0.7255, "recall": 0.8409})
    assert register_training_job(sm, desc, "image:tag", "g") == "arn/new"

    props = sm.calls[0]["CustomerMetadataProperties"]
    assert props["f2"] == "0.8150"
    assert props["training_job"] == "job-1"
    assert sm.calls[0]["ModelApprovalStatus"] == "PendingManualApproval"


def test_register_refuses_when_metric_missing():
    """指標が欠けていたら登録しない。後段の判定を不能にするより手前で止める。"""
    sm = FakeRegistrar()
    desc = _training_desc({"f2": 0.815, "precision": 0.7255})  # recall なし

    with pytest.raises(SystemExit) as e:
        register_training_job(sm, desc, "image:tag", "g")

    assert e.value.code == 2
    assert sm.calls == []