# tests/test_registry.py
"""現行モデルの解決ロジックのテスト。"""

from src.registry import latest_approved


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


def test_returns_newest_approved():
    sm = FakeSageMaker(["arn/3", "arn/1"])
    assert latest_approved(sm, "g")["ModelPackageArn"] == "arn/3"


def test_excludes_candidate_itself():
    """候補自身が Approved でも、比較対象からは外す。"""
    sm = FakeSageMaker(["arn/3", "arn/1"])
    assert latest_approved(sm, "g", exclude_arn="arn/3")["ModelPackageArn"] == "arn/1"


def test_none_when_no_approved():
    assert latest_approved(FakeSageMaker([]), "g") is None