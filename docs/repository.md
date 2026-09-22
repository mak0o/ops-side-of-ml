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

### 基本方針

- 認証情報は `.env` に外出し（`.env.example` を参照）
- ローカルは全ポートを `127.0.0.1` にバインドし LAN に公開しない
- 依存は `requirements.lock` / `requirements-dev.lock` / `requirements-train.lock` / `requirements-serve.lock` で完全固定。CI で全てを pip-audit にかける
- AWS へのアクセスは OIDC のみ。長期のアクセスキーを発行していない
- main はブランチ保護（バイパス無し）。直接 push を禁止し、`lint` / `test` / `scan` / `terraform-fmt` を必須チェックにしている
- GitHub Actions のアクションはコミット SHA で固定し、Dependabot で更新する
- ワークフローのトークンは必要な権限だけを宣言する（既定は `contents: read`）
- コンテナは非 root ユーザーで実行（学習コンテナを除く。理由は [設計判断](decisions.md#学習コンテナは-root推論コンテナは非-root)）
- CI で gitleaks（シークレット検出）と pip-audit（依存の脆弱性）を実行

### 公開リポジトリとしての点検（2026 年 9 月）

アカウントの乗っ取りや情報の露出を懸念して、公開している内容を点検した。

**確認できたこと**

| 懸念 | 結果 |
|---|---|
| 秘密情報の混入 | 全履歴（93 コミット）にアクセスキー、トークン、秘密鍵、`.env`、tfstate は無い |
| メールアドレス | コミットは全て `…@users.noreply.github.com`。SNS の通知先もリポジトリに無い |
| AWS の認証情報 | 長期アクセスキーを発行していない |
| fork からの攻撃 | fork の PR には OIDC トークンが出ないので、AWS のロールを引けない |
| SSO の情報 | SSO の開始 URL や管理者のプロファイル名は掲載していない |
| AWS アカウント ID | ワークフローと tfvars に出ている。ARN に常に含まれる値で秘密情報ではないので許容している |

**最大のリスク**

漏えいではなく構造にある。GitHub アカウントを乗っ取られると、ブランチ保護を外して main に push でき、
apply ロールを通じて AWS の変更権限をほぼ得られる（[apply ロールは実質的な特権ロール](decisions.md#apply-ロールは実質的な特権ロール)）。
対策はここに集中させた。

**実施した対策**

| 対策 | 内容 |
|---|---|
| GitHub アカウントの保護 | 2 要素認証の強化、不要な Personal access token・OAuth アプリ・SSH キーの棚卸し、メールアドレスの非公開設定 |
| GitHub の保護機能 | Secret scanning と Push protection、Dependabot alerts を有効化。外部からの fork の PR はワークフローの実行に承認を必須にした |
| アクションの SHA 固定 | 全アクションを 40 桁のコミット SHA で固定。gitleaks のイメージは `latest` からダイジェスト固定に変更 |
| Dependabot | アクションの更新を週 1 回、1 つの PR にまとめて提案させる |
| スクリプトインジェクションの修正 | plan の出力を PR コメントのスクリプトに直接埋め込んでいたのを、環境変数経由に変更 |
| トークンの権限の最小化 | `ci.yml` / `security.yml` に `contents: read` を宣言。`terraform.yml` は権限をジョブ単位にし、`pull-requests: write` を plan ジョブだけに付けた |
| pip-audit の監査漏れ | `requirements-serve.lock` と `requirements-dev.lock` が監査されていなかったので追加した |
| AWS アカウント | root の MFA とアクセスキーが無いことを確認。IAM Access Analyzer（外部アクセス）と Cost Anomaly Detection を有効化（[詳細](#aws-アカウントの監視)） |

**残っている対策**

| 対策 | 理由 |
|---|---|
| 日常の作業で `AdministratorAccess` を使わない | パイプラインの確認程度なら、権限を絞った権限セットで足りる |
| apply ロールへの Permissions Boundary | ロール名のプレフィクスで絞っても権限昇格を防げていない。練習環境では見送り |
| gitleaks のイメージの更新 | ダイジェストで固定しているが、`run:` の中にあるので Dependabot が追えない。四半期に 1 回、手で確認して更新する |

### AWS アカウントの監視

どちらもアカウント全体の設定なので、このリポジトリの Terraform では管理していない。
プロジェクトを `destroy` したときにアカウントの監視まで消えないようにするためと、
通知先のメールアドレスを公開リポジトリに書かないため（SNS の購読を Terraform の外にしたのと同じ理由）。

**IAM Access Analyzer**

外部アクセスのアナライザー（無料）を東京リージョンに作成した。外部から引き受けられるロールとして 4 件が検出され、全て意図したものだった。

| ロール | 引き受けられる相手 |
|---|---|
| `ops-side-of-ml-gha-plan` / `-apply` / `-push` | GitHub Actions（OIDC） |
| `AWSReservedSSO_AdministratorAccess_…` | IAM Identity Center（SSO でのログインに使うロール） |

S3、ECR、SNS、SageMaker と Step Functions のロールは検出されなかった。外に開いているのは、意図して作った
「GitHub から AWS を操作する経路」と「自分の SSO のログイン経路」だけだった。

4 件はアーカイブルールでアーカイブし、以降に出る検出は全て想定外として扱う。ルールはリソース名と
引き受ける主体（`principal.Federated`）の両方で絞った。リソース名だけで絞ると、そのロールの信頼ポリシーが
書き換えられて別の主体に開いても、自動でアーカイブされて気づけない。

アナライザーには限界がある。**検出結果の `condition` に OIDC の `sub` の条件は表示されない。**
`sub` を `*` に緩めて GitHub 上の誰のリポジトリからでも引き受けられる状態にしても、検出の見た目は変わらない。
この経路を守っているのは PR の plan コメントによるコードレビューとブランチ保護で、アナライザーが見るのは
「GitHub や SSO 以外の想定外の主体に開いていないか」の方。`sub` の条件は IAM から直接確認した。

| ロール | `sub` の条件 |
|---|---|
| `gha-plan` | `repo:mak0o@36266249/ops-side-of-ml@1366131270:*`（`StringLike`） |
| `gha-apply` / `gha-push` | `repo:mak0o@36266249/ops-side-of-ml@1366131270:ref:refs/heads/main`（`StringEquals`） |

**Cost Anomaly Detection**

AWS が作成していた既定のモニター（サービス単位）と通知を使い、閾値だけを変えた。

| 項目 | 変更前 | 変更後 |
|---|---|---|
| 閾値 | 影響額 $100 以上 かつ 影響率 40% 以上 | 影響額 $1 以上 |
| 頻度 | 日次 | 日次 |

普段の請求が月数十円のアカウントに $100 の閾値では、気づいた時点で被害が大きい。影響率の条件は、
$1 の異常なら必ず大きく超えるので外した。

リアルタイムではない。請求データの反映を待つので、乗っ取られて計算資源を悪用された場合でも、
気づくのは早くて数時間後になる。予算アラート（$10）と重ねて使う 2 段目の網という位置づけ。

### リポジトリの外で気をつけること

`aws sso login` の出力には SSO の開始 URL が、各種コマンドの出力にはアカウント ID が含まれる。
ターミナルのログを公開するときは伏せる。開始 URL はフィッシングの材料になる。