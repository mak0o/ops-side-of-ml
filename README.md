# ops-side-of-ml

インフラエンジニアが MLOps を運用側から実装した練習リポジトリ。

題材は **AWS コストの異常検知**（合成データ）。ローカル（MLflow + MinIO）で組んだ
「学習 → 推論 → ドリフト検知 → 再学習 → 昇格判定」のループを、AWS のマネージドサービス
（SageMaker + Step Functions）で組み直し、人手を介さずに一周させている。

作ったものより、**判断とその検証の記録**に重きを置いている。うまくいかなかったことや、
あとから誤りだと分かった結論も消さずに残している。

2026 年 9 月時点で一区切りにしている。対応しなかったことと、その理由は [docs/roadmap.md](docs/roadmap.md#区切り) に記録している。

## できていること

- SageMaker Training Job で学習し、ローカルと同一の評価値を再現
- Batch Transform でバッチ推論し、特徴量込みの推論ログを S3 に出力
- PSI によるドリフト検知を、S3 のデータとモデル同梱のベースラインだけで実行
- SageMaker Model Registry で世代管理（最新の Approved を現行とする）
- 昇格判定：**期間で切ったホールドアウト**で現行と候補を同条件で評価し、下限と**片側 McNemar 検定**で判定
- Step Functions で上記を 1 つのワークフローに。失敗・現行の劣化・モデルの入れ替えを SNS で通知
- Terraform で全リソースを管理。GitHub Actions から OIDC で plan / apply / push（長期キーなし）
- アイドル時の課金ゼロ。月額は S3 と ECR で 10 円前後

## 構成

    Step Functions（EventBridge Scheduler から日次。現在は無効）
      │
      ├─ ResolveModel ──────── Model Registry の最新 Approved を解決
      ├─ Batch Transform ───── 推論ログ (S3, 特徴量 + 予測)
      ├─ Processing Job ────── drift_check ── baseline.json (model.tar.gz から)
      │      └─ ドリフトなし → 終了
      ├─ Training Job ──────── model.tar.gz (S3)
      ├─ Processing Job ────── register_and_promote ── 現行と候補を同じホールドアウトで評価
      │                                                  → Approved / Rejected
      └─ SNS ───────────────── 現行の劣化 / モデルの入れ替えを通知

    EventBridge ルール ─── 実行が FAILED / TIMED_OUT / ABORTED → SNS → メール

ローカル版の構成は [docs/local.md](docs/local.md)、AWS 版の詳細と実行手順は [docs/aws.md](docs/aws.md)。

## 見どころ

### 1. 比較方法を直したら、過去の結論が覆った

当初の昇格判定は、現行と候補を**別々のデータで測った数値**で比べていた。期間で切ったホールドアウトで
同条件に揃えたところ、「旧データと新データを混ぜて学習すると悪化する」という以前の結論が逆転した。
混在データで学習した v6 が、同じホールドアウトで v1 を上回った（F2 0.877 → 0.894）。

→ [docs/promotion.md](docs/promotion.md#昇格判定を同じデータで比べるようにした)

### 2. ブートストラップは何も測っていなかった

v1 → v6 の昇格は 300 行中 3 行の差で決まっていた。ブートストラップで確かめると「片側 95% でほぼ有意」
（差が 0 以下の割合 0.051）に見えたが、これは**食い違った 3 行を 1 つも引かない確率 e^(-3)** にすぎなかった。
判定が食い違った行の偏りを直接見る片側 McNemar 検定（α = 0.05）に切り替え、今はこの規模の差では入れ替わらない。

→ [docs/promotion.md](docs/promotion.md#差が偶然でないかを確かめる)

### 3. 入力のドリフトとモデルの劣化は別物だった

ドリフト検知が反応した特徴量（コストの絶対値）は、モデルの判断材料の 3 割程度だった。
主に頼っていた比率の特徴量は安定しており、現行モデルは下限を割っていなかった。
ただし「劣化していなければ再学習しない」分岐は入れていない。入れていたら v6 の改善を取りこぼしていた。

→ [docs/promotion.md](docs/promotion.md#v1-はドリフト後も下限を割っていなかった)

### 4. 失敗したときに情報が出る作りか

- Serverless Inference はログすら出ずに失敗した。Real-time を一度挟んで「コンテナは正常」を確定させ、仮説を潰した記録を残している（原因は未特定）
- Step Functions の Fail State に固定文言を書き、本当の原因を隠していた
- drift_check がクラッシュすると終了コード 1 になり、「ドリフトあり」と区別できずに再学習が走っていた

→ [docs/serverless.md](docs/serverless.md)、[docs/decisions.md](docs/decisions.md#パイプラインと失敗の扱い)

### 5. IAM を絞っても安全とは限らない

CI の apply ロールは権限をロール名のプレフィクスで絞っているが、それでは権限昇格を防げない。
実際に効いている防御は「main への push でしかロールを引けない」ことと「ブランチ保護」で、
ブランチ保護が外れた瞬間にこの構成は崩れる。

→ [docs/decisions.md](docs/decisions.md#apply-ロールは実質的な特権ロール)

## 技術スタック

| 領域 | 使っているもの |
|---|---|
| ML | scikit-learn（RandomForest）、pandas |
| ローカル | Docker Compose、MLflow、MinIO |
| AWS | SageMaker（Training / Batch Transform / Processing / Model Registry）、Step Functions、S3、ECR、SNS、EventBridge |
| IaC / CI | Terraform（S3 バックエンド、ネイティブロック）、GitHub Actions（OIDC）、ruff、pytest、gitleaks、pip-audit |

## 未解決の課題

- **ラベルがある前提に立っている。** 合成データなので直近の正解ラベルが即座に揃うが、実運用ではラベルは遅れて届く。ホールドアウトによる判定はそのままでは成立しない
- **McNemar は誤報と見逃しを同じ重みで数える。** F2 は見逃しを重く見るので厳密には整合しない
- **推論データを置く上流が無い。** そのため日次スケジュールは無効のまま

→ [docs/roadmap.md](docs/roadmap.md)

## ドキュメント

| ファイル | 内容 |
|---|---|
| [docs/local.md](docs/local.md) | ローカル環境のセットアップ、一周させる手順、ドリフトの再現 |
| [docs/aws.md](docs/aws.md) | AWS での学習・推論・ドリフト検知・昇格・パイプライン・通知 |
| [docs/promotion.md](docs/promotion.md) | 昇格判定の検証（ホールドアウト、結論の見直し、McNemar） |
| [docs/serverless.md](docs/serverless.md) | Serverless Inference の失敗と切り分けの記録 |
| [docs/decisions.md](docs/decisions.md) | 設計判断の一覧（Terraform / CI、SageMaker、パイプライン、通知） |
| [docs/roadmap.md](docs/roadmap.md) | 移行状況、今後、未解決の課題 |
| [docs/repository.md](docs/repository.md) | ファイル構成、テスト、セキュリティ |

## クイックスタート（ローカル）

    cp .env.example .env    # MINIO_ROOT_PASSWORD を設定する
    docker compose up -d
    docker compose run --rm --no-deps ml-app python -m src.generate_data
    docker compose run --rm ml-app python -m src.train --register

続きは [docs/local.md](docs/local.md#一周させる)。
