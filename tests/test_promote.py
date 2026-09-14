"""昇格判定ロジックのテスト。"""

from src.promote import evaluate

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