# リポジトリの構成

## ファイル

| ファイル | 役割 |
|---|---|
| `src/generate_data.py` | 合成データ生成。spike（単発の急増）と creep（削除忘れによる緩やかな増加）の 2 種類の異常を注入する |
| `src/train.py` | 学習、評価、Model Registry への登録、ベースライン統計の保存。SageMaker では `/opt/ml/model` に成果物を書く |
| `src/baseline.py` | PSI 計算のためのビン境界と構成比 |
| `src/serve.py` | 推論 API。ローカルは MLflow の `@production`、SageMaker は `/opt/ml/model/model.joblib` からモデルを読む |
| `src/inference_logger.py` | 推論ログを JSON Lines で S3 にバッファ書き込み |
| `src/drift_check.py` | ベースラインと推論ログを比較して PSI を算出 |
| `src/promote.py` | 候補モデルを現行と比較し、基準を満たせば昇格。MLflow の alias と SageMaker の approval status の両方に対応 |
| `src/registry.py` | SageMaker Model Registry の参照と登録。「現行 = 最新の Approved」の定義をここに集めている |
| `src/register_and_promote.py` | 学習ジョブの成果物を登録し、現行と同じホールドアウトで比べて昇格判定する（Processing Job 用） |
| `src/holdout.py` | ホールドアウトの読み込み、件数の検査、モデルの評価 |
| `src/metrics.py` | 指標の計算。学習時とホールドアウト評価で同じ定義を使うために 1 か所にまとめている |
| `src/results.py` | 判定結果を S3 に JSON で書く（Processing Job 用） |
| `src/replay.py` | データを推論 API に流し込む検証用スクリプト |
| `scripts/retrain_pipeline.sh` | 上記を繋いだ再学習パイプライン |
| `scripts/train_sagemaker.py` | SageMaker Training Job を起動する |
| `scripts/batch_transform.py` | SageMaker Batch Transform を起動する。Model の作成と削除も行う |
| `scripts/prepare_datasets.py` | 時系列データを期間で分け、推論入力・ホールドアウト・再学習データとして S3 に置く |
| `scripts/bootstrap_compare.py` | 現行と候補の差が揺らぎを超えているかを調べる分析用スクリプト（パイプラインからは呼ばない） |
| `infra/` | Terraform。S3 / IAM / ECR / SageMaker Model Registry |
| `infra/pipeline.tf` | Step Functions のステートマシン、そのロール、日次スケジュール（無効） |
| `infra/notifications.tf` | 通知用の SNS トピック、失敗を拾う EventBridge ルール |
| `infra/bootstrap/` | state バケットを作る。ここだけローカル state |
| `docker/` | 用途別の Dockerfile（ml-app / mlflow / train / serve / terraform / aws） |

## テスト

    docker compose run --rm --no-deps ml-app sh -c \
      "pip install -q -r requirements-dev.lock && python -m ruff check . && python -m pytest -q"

PSI 計算、昇格判定ロジック、推論ログの形式の正規化、Model Registry からの現行モデルの解決と登録、
ホールドアウトの検査と評価（学習時の列順で推論すること）、指標の計算、判定が食い違った行の数え方と
片側 McNemar の p 値（v1 → v6 の 3:0 が α=0.05 で通らないこと）をカバーしている。
CI でも lint とあわせて実行される。

Model Registry のテストでは SageMaker の偽物を使い、`list_model_packages` に渡す並び順の引数
（`SortBy="CreationTime"`, `SortOrder="Descending"`）を偽物の側で検査している。
「最新の Approved が現行」という定義はこの引数に依存しているので、誰かが変えたらテストで気づける。

## セキュリティ

- 認証情報は `.env` に外出し（`.env.example` を参照）
- ローカルは全ポートを `127.0.0.1` にバインドし LAN に公開しない
- 依存は `requirements.lock` / `requirements-dev.lock` / `requirements-train.lock` / `requirements-serve.lock` で完全固定。CI で全てを pip-audit にかける
- AWS へのアクセスは OIDC のみ。長期のアクセスキーを発行していない
- main はブランチ保護。直接 push を禁止し、`lint` / `test` / `scan` / `terraform-fmt` を必須チェックにしている
- コンテナは非 root ユーザーで実行（学習コンテナを除く。理由は [設計判断](decisions.md#学習コンテナは-root推論コンテナは非-root)）
- CI で gitleaks（シークレット検出）と pip-audit（依存の脆弱性）を実行
