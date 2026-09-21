import pandas as pd

from src.drift_check import _normalize


def test_nested_features_are_flattened():
    """ローカルの inference_logger 形式（features がネスト）"""
    df = pd.DataFrame([{"prediction": 0, "features": {"cost": 10.0, "cost_ma7": 9.0}}])
    out = _normalize(df)
    assert "features" not in out.columns
    assert out.loc[0, "cost"] == 10.0


def test_flat_features_are_kept():
    """Batch Transform 形式（特徴量が最上位）"""
    df = pd.DataFrame([{"SageMakerOutput": {"is_anomaly": 0}, "cost": 10.0}])
    out = _normalize(df)
    assert out.loc[0, "cost"] == 10.0


def test_mixed_formats_concat():
    """両形式をファイル単位で正規化してから結合できる"""
    nested = pd.DataFrame([{"features": {"cost": 1.0}}])
    flat = pd.DataFrame([{"cost": 2.0}])
    out = pd.concat([_normalize(nested), _normalize(flat)], ignore_index=True)
    assert out["cost"].tolist() == [1.0, 2.0]