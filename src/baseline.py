"""学習データからベースライン統計を計算する。

PSI 計算に必要なビン境界と構成比を保持する。
モデルと同じ run に artifact として紐づけることで、
モデルバージョンとベースラインの対応が崩れないようにする。
"""

import numpy as np
import pandas as pd

N_BINS = 10


def compute_baseline(df: pd.DataFrame, features: list[str]) -> dict:
    """各特徴量のビン境界と構成比を計算する。"""
    stats = {}
    for col in features:
        values = df[col].to_numpy(dtype=float)

        # 分位点でビン境界を作る。両端は無限大にして未知の値も収容する
        quantiles = np.linspace(0, 100, N_BINS + 1)
        edges = np.unique(np.percentile(values, quantiles))
        edges[0], edges[-1] = -np.inf, np.inf

        counts, _ = np.histogram(values, bins=edges)
        ratios = counts / counts.sum()

        stats[col] = {
            "edges": edges.tolist(),
            "ratios": ratios.tolist(),
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    return {"n_samples": len(df), "features": stats}


def psi(baseline: dict, values: np.ndarray) -> float:
    """Population Stability Index を計算する。

    0 除算を避けるため、比率の下限を小さな値でクリップする。
    """
    edges = np.array(baseline["edges"])
    expected = np.array(baseline["ratios"])

    counts, _ = np.histogram(values, bins=edges)
    actual = counts / counts.sum() if counts.sum() > 0 else counts

    eps = 1e-6
    expected = np.clip(expected, eps, None)
    actual = np.clip(actual, eps, None)

    return float(np.sum((actual - expected) * np.log(actual / expected)))


def interpret(score: float) -> str:
    if score < 0.1:
        return "stable"
    if score < 0.25:
        return "moderate"
    return "significant"