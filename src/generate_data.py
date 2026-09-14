"""AWS コストの合成データを生成する。

正常な日次コストに、2種類の異常を注入する:
  - spike: 一時的な急増（設定ミス、想定外のトラフィック）
  - creep: 緩やかな増加（リソースの削除忘れ）
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SERVICES = {
    # service: (基準コスト, 日次のばらつき, 平日/休日の比率)
    "EC2": (120.0, 8.0, 1.4),
    "S3": (45.0, 3.0, 1.05),
    "RDS": (80.0, 4.0, 1.2),
    "Lambda": (15.0, 4.0, 1.6),
    "CloudWatch": (25.0, 2.0, 1.1),
}


def generate(days: int, seed: int, drift: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    rows = []

    for service, (base, noise, weekday_ratio) in SERVICES.items():
        base = base * drift
        costs = np.zeros(days)
        labels = np.zeros(days, dtype=int)

        for i, date in enumerate(dates):
            level = base * (weekday_ratio if date.dayofweek < 5 else 1.0)
            costs[i] = max(0.0, rng.normal(level, noise))

        # spike: 単発の急増
        n_spikes = max(1, days // 60)
        for idx in rng.choice(days, size=n_spikes, replace=False):
            costs[idx] *= rng.uniform(3.0, 8.0)
            labels[idx] = 1

        # creep: 数日かけて増加し、そのまま高止まり
        n_creeps = max(1, days // 120)
        for start in rng.choice(days - 20, size=n_creeps, replace=False):
            length = rng.integers(7, 15)
            ramp = np.linspace(1.0, rng.uniform(1.8, 2.5), length)
            costs[start : start + length] *= ramp
            labels[start : start + length] = 1

        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "service": service,
                    "cost": costs.round(2),
                    "is_anomaly": labels,
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["service", "date"]).copy()
    g = df.groupby("service")["cost"]

    df["cost_ma7"] = g.transform(lambda s: s.rolling(7, min_periods=1).mean())
    df["cost_std7"] = g.transform(lambda s: s.rolling(7, min_periods=1).std()).fillna(0.0)
    df["cost_ratio_ma7"] = df["cost"] / df["cost_ma7"].replace(0, np.nan)
    df["cost_vs_lastweek"] = df["cost"] / g.shift(7).replace(0, np.nan)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    return df.dropna().reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--drift", type=float, default=1.0,
        help="コスト水準の倍率。1.0以外でドリフトしたデータになる",
    )
    parser.add_argument("--out", type=Path, default=Path("data/cost.parquet"))
    parser.add_argument(
        "--append-to", type=Path, default=None,
        help="既存データの続きとして生成し、結合したものを --out に書く",
    )
    args = parser.parse_args()

    raw = generate(args.days, args.seed, args.drift)

    if args.append_to:
        existing = pd.read_parquet(args.append_to)
        start = existing["date"].max() + pd.Timedelta(days=1)
        offset = start - raw["date"].min()
        raw["date"] = raw["date"] + offset
        # 特徴量は結合後の生データから計算する（移動平均が期間をまたぐため）
        base = existing[["date", "service", "cost", "is_anomaly"]]
        raw = pd.concat([base, raw], ignore_index=True)

    df = add_features(raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print(f"{len(df)} rows -> {args.out}")
    print(f"period: {df['date'].min():%Y-%m-%d} .. {df['date'].max():%Y-%m-%d}")
    print(f"anomaly rate: {df['is_anomaly'].mean():.1%}")


if __name__ == "__main__":
    main()