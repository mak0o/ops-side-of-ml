"""指標計算のテスト。"""

from src.metrics import compute_metrics


def test_perfect_prediction():
    m = compute_metrics([1, 0, 1, 0], [1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f2"] == 1.0


def test_values_are_plain_floats():
    """JSON にそのまま書けるよう、numpy の型ではなく float で返す。"""
    m = compute_metrics([1, 0, 1, 0], [1, 1, 0, 0], [0.9, 0.6, 0.4, 0.1])
    assert all(type(v) is float for v in m.values())