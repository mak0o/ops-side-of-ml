# ops-side-of-ml

インフラエンジニアが MLOps を運用側から実装した練習リポジトリ。
題材は **AWS コストの異常検知**。合成データを使い、学習からドリフト検知までのループをローカルで一周させている。

最終目標は同じ構成を AWS (SageMaker) に載せること。ローカルの設計は、そのときの移行コストが最小になるよう選んでいる。

## 構成

    generate_data ──> train ──> Model Registry (@production)
                        │              │
                        │              ├──> serve (推論API)
                        │              │         │
                        ↓              ↓         ↓
                  baseline.json    drift_check <── S3 (推論ログ)

| サービス | 役割 | ポート |
|---|---|---|
| mlflow-server | 実験管理 / Model Registry | 5001 |
| s3 (MinIO) | artifact と推論ログの保存 | 9005 / 9006 |
| ml-app | 学習・推論・バッチの実行環境 | 8000 |

## セットアップ

    cp .env.example .env
    # .env を編集。MINIO_ROOT_PASSWORD にランダムな値を設定する
    #   openssl rand -hex 32

    docker compose up -d

- MLflow UI: http://localhost:5001
- MinIO Console: http://localhost:9006

## 一周させる

    # 1. 合成データを生成
    docker compose run --rm --no-deps ml-app python -m src.generate_data

    # 2. 学習して Model Registry に登録
    docker compose run --rm ml-app python -m src.train --register

    # 3. production alias を付与（初回のみ。以降は昇格したいバージョンを指定）
    docker compose run --rm ml-app python -c "
    import mlflow
    mlflow.MlflowClient().set_registered_model_alias('cost-anomaly-detector', 'production', 1)
    "

    # 4. 推論API を再起動してモデルを読み込む
    docker compose restart ml-app
    curl http://localhost:8000/health

    # 5. 推論を実行（推論ログが S3 に溜まる）
    docker compose run --rm ml-app python -m src.replay --url http://ml-app:8000/predict --limit 200

    # 6. ドリフト検知
    docker compose run --rm ml-app python -m src.drift_check

### ドリフトを再現する

コスト水準を 1.5 倍にしたデータを流し込むと、検知が働くことを確認できる。

    docker compose run --rm --no-deps ml-app python -m src.generate_data \
      --drift 1.5 --seed 99 --out data/cost_drifted.parquet

    docker compose run --rm ml-app python -m src.replay \
      --data data/cost_drifted.parquet --url http://ml-app:8000/predict --limit 400

    docker compose run --rm ml-app python -m src.drift_check

出力例:

    feature                   PSI  status        base_mean  curr_mean
    ------------------------------------------------------------------
    cost                   0.3794  significant       77.87     108.31
    cost_ma7               0.8252  significant       78.28     107.06
    cost_std7              0.1860  moderate          26.42      35.82
    cost_ratio_ma7         0.0255  stable             1.00       1.00
    cost_vs_lastweek       0.0560  stable             1.10       1.10

    DRIFT DETECTED: cost, cost_ma7

絶対値の特徴量 (`cost`, `cost_ma7`) は反応するが、比率の特徴量 (`cost_ratio_ma7`, `cost_vs_lastweek`) は平均が 1.00 のまま動かない。
「利用規模が拡大しただけで、異常の出方そのものは変わっていない」と読める。
この切り分けができると、再学習が必要なのかベースラインの更新で足りるのかを判断できる。

## ファイル

| ファイル | 役割 |
|---|---|
| `src/generate_data.py` | 合成データ生成。spike（単発の急増）と creep（削除忘れによる緩やかな増加）の 2 種類の異常を注入する |
| `src/train.py` | 学習、評価、Model Registry への登録、ベースライン統計の保存 |
| `src/baseline.py` | PSI 計算のためのビン境界と構成比 |
| `src/serve.py` | Model Registry の `@production` からモデルを読み込む推論 API |
| `src/inference_logger.py` | 推論ログを JSON Lines で S3 にバッファ書き込み |
| `src/drift_check.py` | ベースラインと推論ログを比較して PSI を算出 |
| `src/replay.py` | データを推論 API に流し込む検証用スクリプト |

## 設計判断

### 推論ログを MLflow に書かない

推論のたびに `mlflow.start_run()` を呼ぶ実装をよく見かけるが、採用していない。

- MLflow tracking は実験管理用で、推論ログの基盤ではない
- リクエストパスに同期 I/O が入り、MLflow の障害が推論 API の障害になる
- run が際限なく増えてバックエンド DB を圧迫する
- 何より、ドリフト検知に使える形式にならない

代わりに JSON Lines をバッファリングして S3 に書く。パスは Hive 形式のパーティションにしてある。

    s3://mlflow-bucket/inference-logs/year=2026/month=09/day=13/<timestamp>-<uuid>.jsonl

SageMaker Data Capture の出力構造に寄せてあるので、AWS 移行後も `drift_check.py` をほぼそのまま使える。Athena から直接クエリすることもできる。

### ベースラインをモデルと同じ run に置く

ベースライン統計は特定のモデルバージョンと不可分なので、S3 に独立して置くのではなく MLflow の artifact として run に紐づけている。
`@production` alias を辿れば、そのモデルに対応するベースラインが必ず取れる。

### KS 検定ではなく PSI

KS 検定はサンプル数が増えると些細な差でも有意になる。本番では数万件のログが溜まるので、実質的に常に「ドリフトあり」と報告されてしまう。
PSI はサンプル数に依存しにくく、0.1 / 0.25 という実務的な閾値が確立している。

### alias によるモデル切り替え

推論側は `models:/cost-anomaly-detector@production` という固定 URI を参照する。
新しいバージョンを登録しても、alias を付け替えるまで推論側は既存のモデルを使い続ける。昇格もロールバックも alias の操作だけで済み、コードもデプロイ設定も変更しない。

SageMaker Model Registry の approval status が同じ役割を果たすので、この構造のまま移行できる。

### 費用を抑える設計

題材に画像分類ではなく表形式データを選んだのは、SageMaker での費用が桁違いになるため。

| | 画像 (ResNet) | 表形式 |
|---|---|---|
| 学習 | GPU 級 (ml.g4dn.xlarge 〜$0.7/h) | ml.m5.large ($0.13/h)、数秒で完了 |
| 推論 | 常時起動エンドポイントが必要 | Serverless Inference が使える（リクエストがなければ課金ゼロ） |
| イメージ | torch 込みで数 GB | 数百 MB |

コンテナから torch を外したことでイメージサイズは大幅に縮小し、ビルド時間は 137 秒から 26 秒になった。

## セキュリティ

- 認証情報は `.env` に外出し（`.env.example` を参照）
- 全ポートを `127.0.0.1` にバインドし LAN に公開しない
- 依存は `requirements.lock` で完全固定
- コンテナは非 root ユーザーで実行
- CI で gitleaks（シークレット検出）と pip-audit（依存の脆弱性）を実行

## AWS への移行計画

| ローカル | AWS |
|---|---|
| MinIO | S3 |
| MLflow server | SageMaker managed MLflow |
| Model Registry alias | SageMaker Model Registry approval status |
| FastAPI (Docker) | SageMaker Serverless Inference |
| `drift_check.py` | SageMaker Model Monitor、または同じコードを Lambda で実行 |
| 手動実行 | SageMaker Pipelines + EventBridge |

## 今後

- [ ] AWS (SageMaker) への移行
- [ ] ドリフト検知をトリガーとした自動再学習