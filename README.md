# ops-side-of-ml

インフラエンジニアが MLOps を運用側から実装した練習リポジトリ。
題材は **AWS コストの異常検知**。合成データを使い、学習からドリフト検知・再学習判定までのループをローカルで一周させている。

同じ構成を AWS (SageMaker) に載せる作業を進めており、学習と推論は AWS 上で動くところまで到達した。
うまくいかなかった部分もそのまま残している（[Serverless Inference でエンドポイントを作成できなかった](#serverless-inference-でエンドポイントを作成できなかった)）。

## 構成

### ローカル

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

### AWS

    GitHub Actions ──OIDC──> plan / apply / push の 3 ロール
           │
           ├──> ECR (train / serve)
           │
           └──> Terraform ──> S3, IAM, ECR, SageMaker

    train_sagemaker.py ──> SageMaker Training Job ──> model.tar.gz (S3)
                                                          │
                                                          ↓
                                              SageMaker Endpoint ──> 推論ログ (S3)

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

## AWS で学習する

ローカルと同じコードを SageMaker Training Job で実行する。

    docker compose --profile tools run --rm awscli \
      python scripts/train_sagemaker.py --wait

学習データを S3 にアップロードし、ECR のイメージでジョブを起動して完了まで待つ。

    uploaded: s3://ops-side-of-ml-<account>/training-input/<job>/cost.parquet
    started: cost-anomaly-detector-20260920-082722
    InProgress ...
    Completed
    {
      "precision": 0.7255,
      "recall": 0.8409,
      "f1": 0.7789,
      "f2": 0.8150,
      "average_precision": 0.8442
    }
    artifact: s3://.../training-output/<job>/output/model.tar.gz
    billable: 54s

`ml.m5.large` は $0.13/時なので 54 秒で 0.2 円程度。ローカル実行と同じ数値が出る。

成果物の `model.tar.gz` には 4 つのファイルが入る。

| ファイル | 用途 |
|---|---|
| `model.joblib` | モデル本体 |
| `baseline.json` | PSI 計算用のベースライン統計 |
| `features.json` | 特徴量の順序。推論側が列順を再現するために要る |
| `metrics.json` | 学習時の評価指標 |

ローカルでは baseline を MLflow の run に紐づけているが、AWS では同じ tar に入れることで
「モデルとベースラインが不可分」という性質を保っている。

### メトリクスは標準出力から正規表現で拾われる

`src/train.py` の `run_sagemaker()` が `precision=0.7255;` の形式で出力し、
ジョブ定義の `MetricDefinitions` がそれを正規表現で拾って CloudWatch に記録する。

この仕組みには接頭辞の衝突という罠がある。最初 `precision=([0-9\.]+);` と書いたところ、
`average_precision=0.8442;` の行にもマッチして precision が上書きされた。

    "precision": 0.8442        <- average_precision の値が入っている
    "average_precision": 0.8442

f1 と f2 も同様の危険があるので、正規表現は行頭で固定している。

    {"Name": name, "Regex": rf"^{name}=([0-9\.]+);"}

f1 と f2 が正しく出ていたため気づきにくかった。ローカル実行と数値を突き合わせて初めて判明する。

### BillableTimeInSeconds は学習時間ではない

54 秒のうち実際の学習は数秒で、大半はインスタンスの起動とイメージの pull が占める。
学習そのものを速くしても課金は大きく減らない。効くのはイメージサイズとインスタンスタイプの選択。

## AWS で推論する

Terraform で SageMaker Model / Endpoint Configuration / Endpoint を作る。

    docker compose --profile tools run --rm terraform \
      terraform apply -var-file=terraform.ci.tfvars

デプロイするモデルとイメージは `infra/terraform.ci.tfvars` で明示する。

    model_artifact_uri = "s3://.../training-output/<job>/output/model.tar.gz"
    serve_image_tag    = "<commit sha>"

推論の実行。

    docker compose --profile tools run --rm awscli python -c "
    import boto3, json
    rt = boto3.client('sagemaker-runtime', region_name='ap-northeast-1')
    r = rt.invoke_endpoint(
        EndpointName='ops-side-of-ml-endpoint',
        ContentType='application/json',
        Body=json.dumps({
            'cost': 120.5, 'cost_ma7': 78.2, 'cost_std7': 26.4,
            'cost_ratio_ma7': 1.54, 'cost_vs_lastweek': 1.48,
            'day_of_week': 3, 'is_weekend': 0
        }),
    )
    print(r['Body'].read().decode())
    "

    {"is_anomaly":1,"probability":0.890993501731741}

ローカルと同じ値が返る。推論ログも設計どおりのパーティションで S3 に書かれる。

    flushed 1 records -> s3://ops-side-of-ml-<account>/inference-logs/year=2026/month=09/day=20/130052-955e6430.jsonl

このパスは `drift_check.py` がそのまま読める形になっている。

**現在このエンドポイントは destroy してある。** Serverless Inference での作成に失敗し、
切り分けのために立てた Real-time エンドポイント（$0.065/時）を残すと費用方針に反するため。
経緯は次節。

## Serverless Inference でエンドポイントを作成できなかった

同じモデル、同じイメージ、同じ実行ロールで結果が分かれた。

| 構成 | 結果 |
|---|---|
| Real-time (`ml.t2.medium`) | 2分8秒で InService。推論も推論ログの書き出しも正常 |
| Serverless (2048MB / 同時実行 2) | 6分以上待って Failed。3 回試行してすべて同じ |

`FailureReason` は定型文のみ。

    Request to service failed. If failure persists after retry, contact customer support.

**CloudWatch にロググループすら作られない。** つまりコンテナが起動する前に失敗している。
Real-time では同じイメージが次のログを出す。

    INFO:     Started server process [1]
    loaded: version d7256e702e3062edd1d45105df8365bc9273f148
    INFO:     Uvicorn running on http://0.0.0.0:8080
    INFO:     169.254.178.2:47622 - "GET /ping HTTP/1.1" 200 OK

### 潰した仮説

| 仮説 | 確認方法 | 結果 |
|---|---|---|
| イメージがマニフェストリスト（Buildx の attestation 付き）| `ecr batch-get-image` でマニフェストを確認 | 単一の `manifest.v2+json`。該当せず |
| メモリ不足 | `docker stats` で実測 | 起動時 198MB。2048MB に対して 10 倍の余裕 |
| entrypoint が実行できない | ECR のイメージを `docker run <image> serve` で起動 | uvicorn が 8080 で起動。該当せず |
| ECR の pull 権限不足 | 実行ロールのインラインポリシーを確認 | serve リポジトリの ARN が入っている |
| サービスクォータ | `service-quotas list-service-quotas` | 同時実行 10 / エンドポイント 50。上限未達 |
| モデル読み込みの失敗 | `lifespan` で例外を握りつぶさず `raise` するよう変更 | ログに何も出ない = そこまで到達していない |

### 唯一残っている観測

Real-time のログで、SageMaker が `GET /metrics` を叩いて 404 を受け取っている。
Prometheus 形式のメトリクスを取りに来ているが、こちらは実装していない。

    INFO:     169.254.178.2:55314 - "GET /metrics HTTP/1.1" 404 Not Found

Real-time では 404 でも InService になる。Serverless での扱いは不明で、
これが原因だという根拠もない。未検証の仮説として記録しておく。

**原因は未特定。** Real-time で動くことは確認できたので、この構成での Serverless 対応は保留にしている。

### この切り分けから得たもの

失敗したときに情報が出る作りになっているかどうかで、調査時間が桁で変わる。

当初 `src/serve.py` はモデル読み込みに失敗しても起動を続ける実装だった。
ローカルでは合理的（MLflow サーバーが落ちていても `/health` で degraded が分かる）だが、
SageMaker では `/ping` が 503 を返し続け、「エンドポイント作成失敗」としか分からなくなる。

    # SageMaker ではモデルが読めないコンテナに意味がない。
    # 起動を続けると原因の分かりにくい失敗になる。ここで落として理由をログに残す。
    if in_sagemaker():
        raise

同じ理由で、Real-time を一度挟んだのは正解だった。Serverless はログが出ない構成なので、
ログが出る構成で先に「コンテナとイメージは正常」を確定させないと、仮説を潰す先が絞れない。

## ファイル

| ファイル | 役割 |
|---|---|
| `src/generate_data.py` | 合成データ生成。spike（単発の急増）と creep（削除忘れによる緩やかな増加）の 2 種類の異常を注入する |
| `src/train.py` | 学習、評価、Model Registry への登録、ベースライン統計の保存。SageMaker では `/opt/ml/model` に成果物を書く |
| `src/baseline.py` | PSI 計算のためのビン境界と構成比 |
| `src/serve.py` | 推論 API。ローカルは MLflow の `@production`、SageMaker は `/opt/ml/model/model.joblib` からモデルを読む |
| `src/inference_logger.py` | 推論ログを JSON Lines で S3 にバッファ書き込み |
| `src/drift_check.py` | ベースラインと推論ログを比較して PSI を算出 |
| `src/promote.py` | 候補モデルを現行と比較し、基準を満たせば alias を移動 |
| `src/replay.py` | データを推論 API に流し込む検証用スクリプト |
| `scripts/retrain_pipeline.sh` | 上記を繋いだ再学習パイプライン |
| `scripts/train_sagemaker.py` | SageMaker Training Job を起動する |
| `infra/` | Terraform。S3 / IAM / ECR / SageMaker |
| `infra/bootstrap/` | state バケットを作る。ここだけローカル state |
| `docker/` | 用途別の Dockerfile（ml-app / mlflow / train / serve / terraform / aws） |

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

    s3://<bucket>/inference-logs/year=2026/month=09/day=13/<timestamp>-<uuid>.jsonl

SageMaker Data Capture の出力構造に寄せてあるので、AWS 移行後も `drift_check.py` をほぼそのまま使える。Athena から直接クエリすることもできる。

### ベースラインをモデルと同じ run に置く

ベースライン統計は特定のモデルバージョンと不可分なので、S3 に独立して置くのではなく MLflow の artifact として run に紐づけている。
`@production` alias を辿れば、そのモデルに対応するベースラインが必ず取れる。

AWS では MLflow の run が無いので、代わりに `model.tar.gz` の中に同梱している。不可分性という目的は同じ。

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

実測では、学習 1 回あたり 0.2 円（54 秒 × $0.13/時）。
アイドル時に課金されるリソースは無く、現在の月額は S3 と ECR で 10 円程度。

## AWS 側の設計判断

### managed MLflow は使わない

移行表では MLflow server を SageMaker managed MLflow に対応させていたが、採用しない。

トラッキングサーバーは起動している限り課金が続く。Small サイズで us-east-1 が $0.642/時、フランクフルトが $0.886/時。月額にすると $500〜650 になる。torch を外してイメージを削り、表形式データを選んで学習コストを抑えた判断が、これ一つで帳消しになる。

AWS 側では SageMaker Model Registry の approval status を使う。MLflow の alias で設計してあるので読み替えは容易で、追加コストはゼロ。MLflow はローカル専用と割り切る。

### Serverless Inference では Model Monitor が使えない

Serverless Inference は、GPU、VPC 構成、マルチモデルエンドポイント、**データキャプチャ、Model Monitor** が機能除外になっている。

つまりドリフト監視は自前で組む必要がある。推論ログを MLflow に書かず JSON Lines を S3 に吐く設計にしたのは、結果としてこの制約下での唯一の解になっていた。`drift_check.py` は Lambda で実行する。

### 推論ログはバッファせず毎回書く

ローカルでは 10 件ずつバッファして S3 に書いているが、SageMaker では `INFERENCE_LOG_FLUSH_SIZE=1` を渡して毎回書かせる。

Serverless Inference はリクエストが無いとコンテナごと停止する。
停止時に `shutdown` が発火する保証がないので、バッファに残ったログは失われる。
ドリフト検知が推論ログに依存している以上、欠損は許容できない。

これは「推論ログを MLflow に書かない」理由として挙げた同期 I/O の問題を、自ら受け入れることになる。
ただし根拠は違う。

| | MLflow への書き込み | S3 への直接書き込み |
|---|---|---|
| 障害時 | 推論 API の障害になる | 例外を握りつぶせる |
| コスト | run が無限に増える | PUT $0.0047/1000 req |
| 形式 | ドリフト検知に使えない | そのまま使える |

実際、認証情報が無い状態で推論を投げると次のようになる。ログ書き込みは失敗するが推論は 200 を返す。

    inference log upload failed: Unable to locate credentials
    INFO:     "POST /invocations HTTP/1.1" 200 OK

### state バケットを artifacts と分ける

Terraform の state は `ops-side-of-ml-tfstate` に置き、artifacts バケットとは別にしている。同居させない理由は 3 つ。

- artifacts バケットは Terraform の管理対象なので、`destroy` を打つと state ごと消しにいく
- state にはバージョニングが要る（破損時の復旧手段がこれしかない）が、artifacts に付けると推論ログの全世代が残り続けて費用方針と衝突する
- ライフサイクルの prefix を将来広げたとき、state が削除対象に入る

バケットを増やしても課金は増えない。S3 の料金は容量とリクエストに対するもので、バケット数は無料。

### DynamoDB によるロックは使わない

Terraform 1.10 以降、S3 バックエンドは `use_lockfile = true` でロックが取れる。DynamoDB テーブルは不要。

ネット上の記事は大半が DynamoDB 前提のままなので、参照するときは対象バージョンを確認する必要がある。

### bootstrap だけローカル state

`infra/bootstrap/` は state バケット自身を作る構成なので、その state を作成先のバケットに置くことはできない。ここだけローカル state のままにして循環を断っている。

管理対象はバケット 1 つなので、state を失っても `import` で復旧できる。一度作れば以降ほぼ触らない。

### plan と apply でロールを分ける

GitHub Actions からは OIDC でロールを引く。アクセスキーはリポジトリにもシークレットにも置かない。

| ロール | 引ける条件 | 権限 |
| --- | --- | --- |
| `gha-plan` | このリポジトリの任意のワークフロー | ReadOnlyAccess + state バケット書き込み |
| `gha-apply` | main への push のみ | 管理対象リソースの作成・変更 |
| `gha-push` | main への push のみ | ECR への push のみ |

plan ロールにも state への書き込み権限がある。`terraform plan` は refresh で state を更新するため。

plan ロールの `sub` 条件はブランチを限定せず `repo:<repo>:*` にしている。PR は任意のブランチから作られるため。フォークからの PR には `id-token: write` が付与されないので、外部の第三者は引けない。安全性を担保しているのは権限が ReadOnly に限られていることであって、ブランチ条件ではない。

なお `ReadOnlyAccess` は S3 オブジェクトの中身まで読める。現在は合成データと state しかないので許容しているが、実データを置く段階では見直しが要る。

イメージの push に apply ロールを流用せず専用ロールを立てたのは、plan / apply を分けた原則を push にも適用するため。追加コストは Terraform 30 行程度。

### apply ロールは実質的な特権ロール

`gha-apply` の IAM 権限は `ops-side-of-ml-*` というロール名プレフィクスに絞ってあるが、**これは権限昇格を防いでいない**。そのプレフィクスの名前でロールを作り、任意のポリシーをアタッチできるため。

実際に効いている防御は次の 2 つ。

- OIDC の `sub` 条件により main への push でしかロールを引けない
- ブランチ保護により main への直接 push が禁止されている

つまり**ブランチ保護が外れた瞬間にこの構成は崩れる**。IAM を絞ったから安全、という読み方は誤り。

完全に塞ぐには apply を手動承認にするか Permissions Boundary を噛ませる必要があるが、この規模では過剰と判断して採らなかった。

### OIDC の sub には ID が埋め込まれる

信頼ポリシーの `sub` は次の形式になっている。

    repo:mak0o@36266249/ops-side-of-ml@1366131270:ref:refs/heads/main

`@36266249` はユーザー ID、`@1366131270` はリポジトリ ID。GitHub が OIDC トークンに不変 ID を含める設定になっている場合の形式で、リポジトリ名を変更しても条件が壊れない。

多くの記事にある `repo:<owner>/<repo>:...` という形式では一致せず、`Not authorized to perform sts:AssumeRoleWithWebIdentity` になる。
エラーメッセージからは「値が違う」ことしか分からないので、実際に送られている値をワークフロー内でデコードして確認した。

    const token = await core.getIDToken('sts.amazonaws.com');
    const payload = JSON.parse(Buffer.from(token.split('.')[1], 'base64').toString());
    core.info('sub: ' + payload.sub);

### 学習コンテナは root、推論コンテナは非 root

SageMaker は `/opt/ml/model` を root 所有で用意し、学習時はそこにコンテナが書き込む。
非 root では権限エラーになり、Dockerfile 側で chown してもマウント時に上書きされる。

学習ジョブは数分で終了するエフェメラルな実行で、ネットワークにも露出しない。
常時リクエストを受ける推論エンドポイントとはリスクの質が違うので、そちらは非 root を維持する。

推論側の Dockerfile では、`/usr/local/bin` への配置と `chmod` を済ませてから `USER` を切り替える。
順序を逆にすると root 所有ディレクトリへの `COPY` になり、実行ビットの扱いが不安定になる。

### 学習ジョブは Terraform で管理しない

SageMaker Training Job は一度実行して終了するリソースで、Terraform の宣言的管理と噛み合わない。`aws_sagemaker_*` にも Training Job に相当するリソースは無い。

Terraform が持つのは ECR リポジトリと実行ロールまで。ジョブの起動は `boto3` のスクリプトで行う。

エンドポイントは継続的に存在するので Terraform で管理する。
ただし「どのモデルをデプロイするか」は変数で明示し、S3 を検索して最新を自動選択することはしない。
plan の結果が実行時刻に依存すると、plan で見た内容と apply される内容がずれる。

### デプロイするイメージタグに latest を使わない

`serve_image_tag` にはコミットハッシュを入れる。`latest` のままだと、新しいイメージを push しても
Terraform はタグ文字列しか見ないので差分が出ず、古いイメージのまま動き続ける。

そのため「イメージを push する PR」と「タグを更新する PR」の 2 段階になる。
手数は増えるが、どのコミットのイメージが動いているかが state から追える。

## セキュリティ

- 認証情報は `.env` に外出し（`.env.example` を参照）
- ローカルは全ポートを `127.0.0.1` にバインドし LAN に公開しない
- 依存は `requirements.lock` / `requirements-dev.lock` / `requirements-train.lock` / `requirements-serve.lock` で完全固定。CI で全てを pip-audit にかける
- AWS へのアクセスは OIDC のみ。長期のアクセスキーを発行していない
- main はブランチ保護。直接 push を禁止し、`lint` / `test` / `scan` を必須チェックにしている
- コンテナは非 root ユーザーで実行（学習コンテナを除く。理由は上記）
- CI で gitleaks（シークレット検出）と pip-audit（依存の脆弱性）を実行

## AWS への移行状況

| ローカル | AWS | 状態 |
|---|---|---|
| MinIO | S3 | 完了 |
| MLflow server | 使わない（ローカル専用） | 方針確定 |
| Model Registry alias | SageMaker Model Registry approval status | 未着手 |
| `train.py` | SageMaker Training Job | 完了 |
| FastAPI (Docker) | SageMaker Serverless Inference | **失敗**（Real-time では動作確認済み） |
| `drift_check.py` | Lambda（Serverless では Model Monitor が使えない） | 未着手 |
| `retrain_pipeline.sh` | Step Functions + EventBridge | 未着手 |

## 今後

AWS

- [x] Terraform で S3 と IAM を構築、state を S3 バックエンドへ
- [x] GitHub Actions から OIDC で plan / apply / push
- [x] ECR と SageMaker Training Job で学習を AWS に出す
- [x] エンドポイントを立てて推論ログを S3 に溜める（Real-time で確認）
- [ ] Serverless Inference での作成失敗を解決する
- [ ] Model Registry の approval status で昇格を管理する
- [ ] drift_check / promote を Lambda + EventBridge で自動化

その他

- [ ] 判定閾値の最適化とモデルへの記録
- [ ] 学習ウィンドウの自動決定

## 未解決の課題

### 推論ログからラベルを得る手段がない

推論ログには特徴量が記録されているが、**正解ラベルがない**。
予測結果をそのまま正解として再学習すると、モデルが自分の予測を強化するだけになる。

このリポジトリでは合成データを使っているためラベルを生成できるが、実データに移行するとここが最大の障壁になる。
実運用では、検知したアラートを人間が「本当に異常だった / 誤報だった」と判定してラベルを付ける仕組みが必要になる。

AWS に移してもこの問題は解決しない。移行で証明できるのは「同じループをマネージド部品で組み直せる」ことまで。

### 判定閾値が固定

`predict()` の暗黙の閾値 0.5 を使っている。データ分布が変われば最適な閾値も変わるため、本来は検証データで最適化し、モデルと一緒に記録すべき。

実際、全期間データで学習したモデル (v8) は F2 が下がった一方で average_precision は上がっていた。
確率出力のランキングとしては改善しているが、0.5 という切り方が合っていなかったことを示している。

### 学習ウィンドウの設計

`--since` で学習期間を手動指定しているが、実運用では「直近 N 日」のようなルールか、ドリフト検知時点からの自動判定が必要になる。

### 学習データとモデルバージョンの対応

現在 `MODEL_VERSION` には推論イメージのコミットハッシュを入れている。
本来はモデルの世代を示すべきで、イメージのバージョンとは別物。
Model Registry を導入したら Model Package のバージョン番号に置き換える。

### 学習成果物のライフサイクル

`training-input/` と `training-output/` はジョブごとにプレフィクスが増え続ける。
現在ライフサイクルルールの対象は `inference-logs/` のみなので、こちらは無期限に残る。
1 回あたり数百 KB なので当面は問題にならないが、再学習を自動化すると増加が早まる。

## 補足

### MinIO のイメージ取得元

MinIO は 2026 年 9 月に Docker Hub から `minio/minio` と `minio/mc` を削除した。
Docker Hub は匿名 pull に 401 を返すため `pull access denied ... may require 'docker login'` という認証エラー風のメッセージが出るが、認証の問題ではなくリポジトリ自体が存在しない。

現在の公式配布元は quay.io。

    image: quay.io/minio/minio:RELEASE.2025-09-07T16-13-09Z
    image: quay.io/minio/mc

### コンテナごとに依存を分ける

用途別に lock ファイルを分けている。

|  | ローカル実行環境 | 学習コンテナ | 推論コンテナ |
| --- | --- | --- | --- |
| 直接依存 | 9 | 4 | 7 |
| lock のパッケージ数 | 294 行 | 11 | - |
| 転送サイズ | 334 MB | 193 MB | 168 MB |

削減の狙いは保管費用ではない。ECR は $0.10/GB/月なので、140 MB 減らしても月 1 円台にしかならない。
効くのはジョブ起動時の pull 時間と CI のビルド時間で、ジョブが数分で終わる規模ではそこも支配的ではない。

それでも分けたのは、`mlflow` を学習・推論コンテナに入れると AWS 側で使わないライブラリを運ぶことになり、
「AWS では MLflow を使わない」という判断とイメージの中身が食い違うため。
`src/train.py` と `src/serve.py` では `mlflow` の import をローカル用の関数の内側に移してある。

ただし推論コンテナから mlflow を外したことで、SageMaker 環境の判定に失敗すると
`No module named 'mlflow'` で落ちるようになった。判定は `/opt/ml/model/model.joblib` の存在で行っている。

### 学習ジョブの実行に使い捨てコンテナを使う

macOS の Homebrew Python は PEP 668 で保護されているため、ホストに boto3 を入れられない。
`docker compose --profile tools` に `awscli` サービスを用意し、`~/.aws` をマウントして実行している。

    docker compose --profile tools run --rm awscli python scripts/train_sagemaker.py --wait

SSO トークンの更新でコンテナが書き込みを行うため、`~/.aws/sso` は read-only でマウントしない。
それ以外（`config` / `credentials`）は read-only のまま。