"""昇格判定ロジックのテスト。"""

from src.promote import _metrics_from_package, evaluate, mcnemar_p, significance_check

GOOD = {"f2": 0.833, "precision": 0.778, "recall": 0.848}


def passed(checks) -> bool:
    return all(ok for _, ok, _ in checks)


def test_first_promotion_passes_without_current():
    assert passed(evaluate(None, GOOD, 0.6, 0.8))


def test_improvement_passes():
    candidate = {"f2": 0.850, "precision": 0.780, "recall": 0.870}
    assert passed(evaluate(GOOD, candidate, 0.6, 0.8))


def test_identical_performance_is_rejected():
    """改善がないなら入れ替えるリスクを取る理由がない。"""
    assert not passed(evaluate(GOOD, dict(GOOD), 0.6, 0.8))


def test_degradation_is_rejected():
    candidate = {"f2": 0.787, "precision": 0.609, "recall": 0.848}
    assert not passed(evaluate(GOOD, candidate, 0.6, 0.8))


def test_precision_floor_blocks_high_recall_model():
    """recall を上げて f2 が改善しても、precision が下限割れなら拒否する。"""
    candidate = {"f2": 0.900, "precision": 0.40, "recall": 0.99}
    checks = evaluate(GOOD, candidate, 0.6, 0.8)
    assert not passed(checks)
    assert dict((n, ok) for n, ok, _ in checks)["precision"] is False


def test_recall_floor_blocks_low_recall_model():
    candidate = {"f2": 0.840, "precision": 0.95, "recall": 0.70}
    checks = evaluate(GOOD, candidate, 0.6, 0.8)
    assert not passed(checks)
    assert dict((n, ok) for n, ok, _ in checks)["recall"] is False


def test_package_metadata_is_converted_to_float():
    """SageMaker のメタデータは文字列なので float に戻し、数値でない値は捨てる。"""
    desc = {"CustomerMetadataProperties": {
        "f2": "0.8150", "precision": "0.7255", "training_job": "job-123",
    }}
    metrics = _metrics_from_package(desc)
    assert metrics == {"f2": 0.815, "precision": 0.7255}




def test_mcnemar_one_sided_p_values():
    """片側の正確な McNemar。二項分布の裾の和。"""
    assert mcnemar_p(3, 0) == 0.125
    assert mcnemar_p(5, 0) == 1 / 32
    assert mcnemar_p(7, 1) == 9 / 256
    assert mcnemar_p(0, 0) == 1.0


def test_three_rows_are_not_enough():
    """v1 と v6 の実例。誤報 3 件の差では、α=0.05 で昇格しない。"""
    name, ok, _ = significance_check(3, 0, alpha=0.05)
    assert name == "mcnemar"
    assert not ok


def test_five_one_sided_rows_are_enough():
    _, ok, _ = significance_check(5, 0, alpha=0.05)
    assert ok