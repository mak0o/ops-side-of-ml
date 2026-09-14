"""ベースライン統計と PSI 計算のテスト。"""

import numpy as np
import pandas as pd
import pytest

from src.baseline import compute_baseline, interpret, psi


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "cost": rng.normal(100, 10, 1000),
        "day_of_week": rng.integers(0, 7, 1000),
    })


def test_compute_baseline_structure(sample_df):
    b = compute_baseline(sample_df, ["cost", "day_of_week"])

    assert b["n_samples"] == 1000
    assert set(b["features"]) == {"cost", "day_of_week"}

    stats = b["features"]["cost"]
    assert {"edges", "ratios", "mean", "std"} <= set(stats)
    # 構成比の合計は 1
    assert sum(stats["ratios"]) == pytest.approx(1.0)
    # 両端は無限大なので未知の値も収容できる
    assert stats["edges"][0] == -np.inf
    assert stats["edges"][-1] == np.inf


def test_psi_is_zero_for_identical_distribution(sample_df):
    b = compute_baseline(sample_df, ["cost"])
    score = psi(b["features"]["cost"], sample_df["cost"].to_numpy())
    assert score == pytest.approx(0.0, abs=1e-6)


def test_psi_detects_shift(sample_df):
    b = compute_baseline(sample_df, ["cost"])
    shifted = sample_df["cost"].to_numpy() * 1.5
    score = psi(b["features"]["cost"], shifted)
    assert score > 0.25


def test_psi_handles_out_of_range_values(sample_df):
    """ベースラインの範囲外の値でも例外にならず、高い PSI を返す。"""
    b = compute_baseline(sample_df, ["cost"])
    extreme = np.full(100, 10_000.0)
    score = psi(b["features"]["cost"], extreme)
    assert np.isfinite(score)
    assert score > 0.25


def test_interpret_thresholds():
    assert interpret(0.05) == "stable"
    assert interpret(0.15) == "moderate"
    assert interpret(0.30) == "significant"
    # 境界値
    assert interpret(0.1) == "moderate"
    assert interpret(0.25) == "significant"