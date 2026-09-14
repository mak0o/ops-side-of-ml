"""コスト異常検知モデルを学習し、MLflow Model Registry に登録する。"""

import argparse
import json
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.baseline import compute_baseline

FEATURES = [
    "cost",
    "cost_ma7",
    "cost_std7",
    "cost_ratio_ma7",
    "cost_vs_lastweek",
    "day_of_week",
    "is_weekend",
]
TARGET = "is_anomaly"
MODEL_NAME = "cost-anomaly-detector"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/cost.parquet"))
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--register", action="store_true",
                        help="Model Registry に登録する")
    parser.add_argument("--since", type=str, default=None,
                        help="この日付以降のデータのみで学習する（例: 2025-11-01）")
    args = parser.parse_args()

    df = pd.read_parquet(args.data)

    if args.since:
        before = len(df)
        df = df[df["date"] >= pd.Timestamp(args.since)]
        print(f"window: {args.since} 以降に絞り込み ({before} -> {len(df)} rows)")
        if df.empty:
            raise SystemExit("指定期間にデータがありません")

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42, stratify=df[TARGET]
    )

    mlflow.set_experiment("cost-anomaly-detection")

    with mlflow.start_run() as run:
        mlflow.log_params({
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "n_train": len(X_train),
            "anomaly_rate": float(df[TARGET].mean()),
            "since": args.since or "all",
        })

        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        metrics = {
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "f2": fbeta_score(y_test, y_pred, beta=2, zero_division=0),
            "average_precision": average_precision_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)

        baseline = compute_baseline(X_train, FEATURES)
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/baseline.json"
            with open(path, "w") as f:
                json.dump(baseline, f, indent=2)
            mlflow.log_artifact(path)

        mlflow.sklearn.log_model(
            clf,
            name="model",
            input_example=X_train.head(3),
            registered_model_name=MODEL_NAME if args.register else None,
            pip_requirements=["scikit-learn", "pandas", "mlflow"],
        )

        print(f"run_id: {run.info.run_id}")
        for k, v in metrics.items():
            print(f"{k}: {v:.3f}")


if __name__ == "__main__":
    main()