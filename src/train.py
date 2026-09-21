"""コスト異常検知モデルを学習する。

ローカル: MLflow に記録し Model Registry に登録する。
SageMaker: /opt/ml/model に成果物を書き出す。回収と登録は SageMaker 側の仕事。
"""

import argparse
import json
import tempfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.baseline import compute_baseline
from src.metrics import compute_metrics

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

# SageMaker Training Job の規約パス
SM_CONFIG = Path("/opt/ml/input/config/hyperparameters.json")
SM_INPUT = Path("/opt/ml/input/data/train")
SM_MODEL = Path("/opt/ml/model")


def in_sagemaker() -> bool:
    """SageMaker Training Job 内で動いているか。

    SageMaker はコンテナ起動時に必ずこのファイルを書く。
    環境変数は実行方式によって有無が変わるため、ファイルの存在で判定する。
    """
    return SM_CONFIG.exists()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/cost.parquet"))
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--register", action="store_true",
                        help="Model Registry に登録する（ローカルのみ）")
    parser.add_argument("--since", type=str, default=None,
                        help="この日付以降のデータのみで学習する（例: 2025-11-01）")
    args = parser.parse_args()

    if in_sagemaker():
        args = _apply_sagemaker_config(args)

    return args


def _apply_sagemaker_config(args: argparse.Namespace) -> argparse.Namespace:
    """SageMaker はハイパーパラメータを JSON の文字列値として渡してくる。

    コマンドライン引数では渡らないので、ここで上書きする。
    """
    hp = json.loads(SM_CONFIG.read_text())

    if "n_estimators" in hp:
        args.n_estimators = int(hp["n_estimators"])
    if "max_depth" in hp:
        args.max_depth = int(hp["max_depth"])
    if hp.get("since"):
        args.since = hp["since"]

    # 入力データは train チャネルのディレクトリに展開される
    files = sorted(SM_INPUT.glob("*.parquet"))
    if not files:
        raise SystemExit(f"{SM_INPUT} に parquet がありません")
    args.data = files[0]

    return args


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_parquet(args.data)

    if args.since:
        before = len(df)
        df = df[df["date"] >= pd.Timestamp(args.since)]
        print(f"window: {args.since} 以降に絞り込み ({before} -> {len(df)} rows)")
        if df.empty:
            raise SystemExit("指定期間にデータがありません")

    return df


def train(df: pd.DataFrame, args: argparse.Namespace):
    """学習と評価。実行環境に依存しない部分。"""
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42, stratify=df[TARGET]
    )

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # ランダム分割なので、移動平均の特徴量を通じて隣接日の情報が漏れる。
    # この値は楽観的で、昇格判定の比較には使わない（ホールドアウトで評価し直す）。
    metrics = compute_metrics(y_test, y_pred, y_proba)


    baseline = compute_baseline(X_train, FEATURES)

    return clf, metrics, baseline, X_train


def run_local(df: pd.DataFrame, args: argparse.Namespace) -> None:
    # mlflow はローカルでしか使わない。SageMaker 用イメージに含めないため
    # トップレベルではなくここで import する。
    import mlflow
    import mlflow.sklearn

    mlflow.set_experiment("cost-anomaly-detection")

    with mlflow.start_run() as run:
        mlflow.log_params({
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "n_train": int(len(df) * 0.8),
            "anomaly_rate": float(df[TARGET].mean()),
            "since": args.since or "all",
        })

        clf, metrics, baseline, X_train = train(df, args)
        mlflow.log_metrics(metrics)

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


def run_sagemaker(df: pd.DataFrame, args: argparse.Namespace) -> None:
    """成果物を /opt/ml/model に書く。SageMaker が tar にまとめて S3 に置く。"""
    clf, metrics, baseline, _ = train(df, args)

    SM_MODEL.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, SM_MODEL / "model.joblib")

    # ベースラインと特徴量順をモデルと同じ tar に入れる。
    # ローカルで MLflow の run に紐づけているのと同じ意図。
    (SM_MODEL / "baseline.json").write_text(json.dumps(baseline, indent=2))
    (SM_MODEL / "features.json").write_text(json.dumps(FEATURES, indent=2))
    (SM_MODEL / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # SageMaker の MetricDefinitions がこの形式を正規表現で拾い、
    # CloudWatch と Training Job のメタデータに記録する。
    for k, v in metrics.items():
        print(f"{k}={v:.4f};")


def main() -> None:
    args = parse_args()
    df = load_data(args)

    if in_sagemaker():
        run_sagemaker(df, args)
    else:
        run_local(df, args)


if __name__ == "__main__":
    main()