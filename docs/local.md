# ローカルで一周させる

MLflow と MinIO を Docker Compose で立て、学習からドリフト検知・昇格判定までをローカルで回す。
AWS 版の元になった構成。

## 構成

    generate_data ──> train ──> Model Registry (@production)
                        │              │
                        │              ├──> serve (推論API)
                        │              │         │
                        ↓              ↓         ↓
                  baseline.json    drift_check <── S3 (推論ログ)
                                       │
                                       ↓
                                   promote (昇格判定)

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

## 再学習パイプライン

ドリフト検知から昇格判定までを 1 コマンドで実行する。

    ./scripts/retrain_pipeline.sh

    # 判定のみ行い alias を動かさない
    DRY_RUN=1 ./scripts/retrain_pipeline.sh

流れは次の通り。

1. `drift_check` を実行。ドリフトがなければここで終了
2. 検知した場合は再学習して Model Registry に登録
3. 最新バージョンを取得
4. `promote` で昇格判定。基準を満たせば alias を移動し、推論 API を再起動

### 再学習は必ずしも改善しない

> **この節の比較には欠陥がある。** 表の数値は各モデルが別々のデータで測った値で、比較可能ではない。
> 同じホールドアウトで比べ直したところ、「旧+新の混在で学習すると悪化する」は逆の結果になった。
> 経緯は [過去の結論の見直し](promotion.md#過去の結論の見直し) を参照。当初の記述はそのまま残している。

ドリフトを検知して再学習しても、性能が上がるとは限らない。実際に試した結果:

| 学習データ | precision | recall | F2 | 判定 |
|---|---|---|---|---|
| 現行 (v4) | 0.778 | 0.848 | 0.833 | - |
| 全期間（旧+新の混在） | 0.725 | 0.841 | 0.815 | REJECT |
| 直近のみ（600 行に絞込） | 0.889 | 0.667 | 0.702 | REJECT |

混在させると 2 つの分布の中間を学んでしまい、絞り込むとサンプル不足で recall が落ちる。
異常検知は正例が少ないタスクなので、学習データを削ると真っ先に見逃しが増える。

いずれも昇格ゲートが劣化を検知して拒否した。
「ドリフトを検知したから再学習する、再学習したから新しいモデルを使う」と自動化していたら、性能の落ちたモデルが本番に出ていた。
判定を挟む設計の価値がここにある。
