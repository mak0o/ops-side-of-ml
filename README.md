# ops-side-of-ml

インフラエンジニアが MLOps を運用側から実装した練習リポジトリ。
題材は **AWS コストの異常検知**。合成データを使い、学習からドリフト検知・再学習判定までのループをローカルで一周させている。

最終目標は同じ構成を AWS (SageMaker) に載せること。ローカルの設計は、そのときの移行コストが最小になるよう選んでいる。

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

## ファイル

| ファイル | 役割 |
|---|---|
| `src/generate_data.py` | 合成データ生成。spike（単発の急増）と creep（削除忘れによる緩やかな増加）の 2 種類の異常を注入する |
| `src/train.py` | 学習、評価、Model Registry への登録、ベースライン統計の保存 |
| `src/baseline.py` | PSI 計算のためのビン境界と構成比 |
| `src/serve.py` | Model Registry の `@production` からモデルを読み込む推論 API |
| `src/inference_logger.py` | 推論ログを JSON Lines で S3 にバッファ書き込み |
| `src/drift_check.py` | ベースラインと推論ログを比較して PSI を算出 |
| `src/promote.py` | 候補モデルを現行と比較し、基準を満たせば alias を移動 |
| `src/replay.py` | データを推論 API に流し込む検証用スクリプト |
| `scripts/retrain_pipeline.sh` | 上記を繋いだ再学習パイプライン |

## テスト

    docker compose run --rm --no-deps ml-app sh -c \
      "pip install -q -r requirements-dev.lock && python -m ruff check . && python -m pytest -q"

PSI 計算と昇格判定ロジックをカバーしている。CI でも lint とあわせて実行される。

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

### 自動昇格の判定基準

`src/promote.py` は候補モデルを現行モデルと比較し、基準を満たしたときだけ `production` alias を移動する。

主指標は **F2**（recall を precision の 2 倍重視）。
コスト異常検知では見逃し（リソース削除忘れに気づかず請求が発生する）のほうが、誤検知（不要な調査に数分使う）より損失が大きいため。

ただし F2 だけで判断すると、precision が崩壊しても recall さえ上がれば「改善」と判定されうる。
そのため precision >= 0.60、recall >= 0.80 の下限を別途設けている。

    current  (v4): f2=0.833  precision=0.778  recall=0.848
    candidate(v5): f2=0.787  precision=0.609  recall=0.848

    f2           0.833 -> 0.787 (-0.047)        FAIL
    precision    0.609 >= 0.60                  PASS
    recall       0.848 >= 0.80                  PASS

    REJECT: version 5 は昇格基準を満たしません

終了コードは 0（昇格可）/ 1（基準未達）/ 2（判定不能）で分けてあり、パイプラインから条件分岐に使える。

指標が欠損している場合はエラーで停止する。欠損を 0 として扱うと、新しい指標を追加した直後にすべての候補が無条件で昇格してしまう。
判断材料が揃わないときは止まる、を原則にしている。

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

## 未解決の課題

### 推論ログからラベルを得る手段がない

推論ログには特徴量が記録されているが、**正解ラベルがない**。
予測結果をそのまま正解として再学習すると、モデルが自分の予測を強化するだけになる。

このリポジトリでは合成データを使っているためラベルを生成できるが、実データに移行するとここが最大の障壁になる。
実運用では、検知したアラートを人間が「本当に異常だった / 誤報だった」と判定してラベルを付ける仕組みが必要になる。

### 判定閾値が固定

`predict()` の暗黙の閾値 0.5 を使っている。データ分布が変われば最適な閾値も変わるため、本来は検証データで最適化し、モデルと一緒に記録すべき。

実際、全期間データで学習したモデル (v8) は F2 が下がった一方で average_precision は上がっていた。
確率出力のランキングとしては改善しているが、0.5 という切り方が合っていなかったことを示している。

### 学習ウィンドウの設計

`--since` で学習期間を手動指定しているが、実運用では「直近 N 日」のようなルールか、ドリフト検知時点からの自動判定が必要になる。

## セキュリティ

- 認証情報は `.env` に外出し（`.env.example` を参照）
- 全ポートを `127.0.0.1` にバインドし LAN に公開しない
- 依存は `requirements.lock` / `requirements-dev.lock` で完全固定
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
| `retrain_pipeline.sh` | SageMaker Pipelines + EventBridge |

## 今後

- [ ] AWS (SageMaker) への移行
- [ ] 判定閾値の最適化とモデルへの記録
- [ ] 学習ウィンドウの自動決定

## 補足

### MinIO のイメージ取得元

MinIO は 2026 年 9 月に Docker Hub から `minio/minio` と `minio/mc` を削除した。
Docker Hub は匿名 pull に 401 を返すため `pull access denied ... may require 'docker login'` という認証エラー風のメッセージが出るが、認証の問題ではなくリポジトリ自体が存在しない。

現在の公式配布元は quay.io。

    image: quay.io/minio/minio:RELEASE.2025-09-07T16-13-09Z
    image: quay.io/minio/mc