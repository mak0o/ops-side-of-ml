"""ホールドアウト評価のテスト。"""

import numpy as np
import pandas as pd
import pytest

from src.holdout import check_holdout, score


class RecordingModel:
    """渡された列の順序を記録し、cost > 100 を異常と判定する偽物。"""

    def __init__(self):
        self.columns = None

    def predict(self, X):
        self.columns = list(X.columns)
        return (X["cost"] > 100).astype(int).to_numpy()

    def predict_proba(self, X):
        p = (X["cost"] > 100).astype(float).to_numpy()
        return np.column_stack([1 - p, p])


def _holdout(positives: int, negatives: int) -> pd.DataFrame:
    return pd.DataFrame({
        "cost": [150.0] * positives + [50.0] * negatives,
        "day_of_week": [1] * (positives + negatives),
        "extra": [0] * (positives + negatives),
        "is_anomaly": [1] * positives + [0] * negatives,
    })


def test_score_uses_feature_order_from_artifact():
    """推論時の列順は学習時の features.json に従う。余分な列は渡さない。"""
    model = RecordingModel()
    score(model, ["day_of_week", "cost"], _holdout(3, 3))
    assert model.columns == ["day_of_week", "cost"]


def test_refuses_too_few_anomalies():
    with pytest.raises(SystemExit) as e:
        check_holdout(_holdout(positives=3, negatives=50), min_positives=10)
    assert e.value.code == 2


def test_refuses_holdout_without_labels():
    df = _holdout(20, 20).drop(columns=["is_anomaly"])
    with pytest.raises(SystemExit) as e:
        check_holdout(df)
    assert e.value.code == 2