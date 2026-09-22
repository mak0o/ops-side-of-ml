# 設計判断

判断ごとに、何を選び、何を選ばなかったか、その理由を残している。
実際に踏んで直したものも含む。

- [ML とローカル環境](#ml-とローカル環境)
- [Terraform と CI/CD](#terraform-と-cicd)
- [SageMaker](#sagemaker)
- [パイプラインと失敗の扱い](#パイプラインと失敗の扱い)
- [通知](#通知)
- [補足](#補足)

## ML とローカル環境

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
PSI は件数が多いときにこの過敏さが出にくく、0.1 / 0.25 という実務的な閾値が確立している。

ただし逆に、**件数が少ないと PSI は不安定になる**。同じ母集団から取った 50 件で significant が出た
（[最初の検証は間違った結論を出していた](aws.md#最初の検証は間違った結論を出していた)）。
そのため `drift_check` は件数が下限を下回ると判定不能として止まる。

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

SageMaker では Model Registry の approval status が同じ役割を果たす。
実際にこの構造のまま移行でき、判定ロジックは一行も変えずに済んだ。

### 費用を抑える設計

題材に画像分類ではなく表形式データを選んだのは、SageMaker での費用が桁違いになるため。

| | 画像 (ResNet) | 表形式 |
|---|---|---|
| 学習 | GPU 級 (ml.g4dn.xlarge 〜$0.7/h) | ml.m5.large ($0.13/h)、数秒で完了 |
| 推論 | 常時起動エンドポイントが必要 | Serverless Inference が使える（リクエストがなければ課金ゼロ） |
| イメージ | torch 込みで数 GB | 数百 MB |

コンテナから torch を外したことでイメージサイズは大幅に縮小し、ビルド時間は 137 秒から 26 秒になった。

実測では、学習 1 回あたり 0.2 円（54 秒 × $0.13/時）。バッチ推論は課金秒数が 1 秒未満。
パイプラインを 1 回回すと、再学習まで進んでも数円。
アイドル時に課金されるリソースは無く、現在の月額は S3 と ECR で 10 円程度。

日次スケジュールを有効にすると、ドリフトが無い日でも Transform と Processing Job で 1 日数円、
月 100 円前後になる。スケジュールは Terraform で作ってあるが、推論データを `latest/` に置く上流が無いので無効にしている。
有効にすると毎日データ無しで失敗し、失敗通知が届き続ける。

SNS のメール通知と EventBridge の AWS サービスイベントは、この規模では無料枠に収まる。

## Terraform と CI/CD

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

### 権限の追加とリソースの作成を別の PR にする

apply ロールに `states:*` と `scheduler:*` を足す変更と、ステートマシンを作る変更は PR を分けた。

同じ PR にすると、Terraform は依存関係の無い両者を並列に作ろうとし、権限が反映される前に
ステートマシンの作成が走って失敗する。CI で apply するロールの権限を広げるときは、先にそれだけを適用する。

### 必須でないチェックは実質的に無い

`terraform fmt -check` は、当初 `terraform.yml` の plan ジョブの中にだけあった。plan ジョブは `infra/` の
変更でしか起動しない（paths フィルタ）ので必須チェックにできず、落ちても PR はマージできた。

実際に、書式の崩れた `pipeline.tf` を含む PR がマージされかけた。原因は、属性の間にコメントを挟んだこと。

    statement {
      actions = ["states:StartExecution"]
      # ステートマシンの属性を参照すると、定義を変えるたびに plan でこのポリシーも
      # 変更扱いになる。名前は固定なので ARN を組み立てて依存を切る。
      resources = ["arn:aws:states:..."]
    }

`terraform fmt` は連続した属性の `=` の位置を揃える。コメントを挟むと 2 つの属性は連続していない扱いになり、
`actions   =` の揃え用の空白が不要になる。それが `fmt -check` で差分として検出された。

`fmt -check` は AWS の認証が要らないので、paths フィルタの無い `ci.yml` に `terraform-fmt` ジョブとして移し、
必須チェックに加えた。必須でないチェックは、落ちていても誰も止めないので、運用上は無いのと同じだった。

### デプロイするイメージタグに latest を使わない

`serve_image_tag` にはコミットハッシュを入れる。`latest` のままだと、新しいイメージを push しても
Terraform はタグ文字列しか見ないので差分が出ず、古いイメージのまま動き続ける。

そのため「イメージを push する PR」と「タグを更新する PR」の 2 段階になる。
手数は増えるが、どのコミットのイメージが動いているかが state から追える。

### アクションはコミット SHA で固定する

`aws-actions/configure-aws-credentials@v5` のようなタグ指定は、アクションのリポジトリが乗っ取られると
タグを付け替えられる。このリポジトリでは、そのアクションが OIDC で得た apply ロールの認証情報を扱うので、
付け替えられたコードが認証情報を持ち出せる。

全アクションを 40 桁のコミット SHA で固定し、末尾にバージョンをコメントで残した。

    - uses: aws-actions/configure-aws-credentials@61815dcd50bd041e203e49132bacad1fd04d2708 # v5.1.1

固定した時点で、メジャータグ（`v5` など）と最新のパッチ版タグは同じコミットを指していたので、挙動は変わっていない。
更新は Dependabot に任せ、SHA とコメントをまとめて書き換えた PR を受け取る。

最初は全ての更新を 1 つの PR にまとめる設定にしていた。最初に届いた PR は 5 つともメジャー版の更新で、
うち 2 つ（checkout と github-script）は 2 段飛ばしだった。マージの前に、5 つ分の SHA が公式のタグと一致するかと、
互換性の無い変更がこのリポジトリの使い方に影響しないかを、全てのリリースノートで確かめる必要があった。

マイナー・パッチ版とメジャー版でグループを分けた。マイナー・パッチ版は 1 つの PR にまとまり、
メジャー版はアクションごとの PR として届く。リリースノートの確認はメジャー版の PR だけで済む。

Dependabot が作った PR では、fork からの PR と同じく OIDC のトークンが発行されない。そのため
`terraform / plan` は AWS のロールを引けずに失敗し、`configure-aws-credentials` や `setup-terraform` の
新しい版が実際に AWS に接続するのはマージ後の apply と image のジョブが初めてになる。
メジャー版をマージした直後は、この 2 つのジョブの結果を確認する。

gitleaks は `docker run` で呼んでいるので Dependabot が追えない。`latest` からイメージのダイジェストでの固定にし、四半期に 1 回、手で更新する。タグ（`v8.30.1`）で固定するだけでは、公開者のアカウントが乗っ取られると同じ名前のまま中身を差し替えられる。ダイジェストはマルチアーキテクチャのインデックスの値を使い、手元の arm64 と GitHub のランナーの amd64 の両方で同じ値が解決されることを確かめた。

### ワークフローに外部の出力を直接埋め込まない

PR に plan の結果をコメントするステップで、plan の出力を JavaScript のテンプレート文字列に直接埋め込んでいた。

    const output = `... ${{ steps.plan.outputs.stdout }} ...`;

`${{ }}` はスクリプトが実行される前に文字列として展開されるので、plan の出力に `` ` `` や `${` が含まれると、
それがコードとして実行される。Terraform の設定に書いた文字列（説明文やタグの値）は plan の出力に現れる。

環境変数で渡し、スクリプトの中では `process.env.PLAN` として文字列で扱うように直した。
書き込み権限を持つのが自分だけなので悪用は難しかったが、典型的な脆弱パターン。

### トークンの権限はジョブ単位で宣言する

`ci.yml` と `security.yml` には `permissions` が無く、リポジトリの既定の権限でトークンが発行されていた。
`contents: read` を明示した。

`terraform.yml` はワークフロー全体に `pull-requests: write` を付けていたので、PR にコメントしない apply ジョブにも
その権限があった。権限をジョブ単位に移し、plan ジョブだけに付けた。

### 監査していると書いた lock が監査されていなかった

README には「4 つの lock を全て pip-audit にかける」と書いていたが、実際に監査していたのは
`requirements.lock` と `requirements-train.lock` の 2 つだけだった。推論イメージ用の `requirements-serve.lock` が
漏れており、Processing Job もそのイメージで動いていた。

lock を後から増やしたときに、監査の設定を更新し忘れていた。セキュリティの点検で、記述と実態を突き合わせて見つかった。
追加した結果、脆弱性は検出されなかった。

## SageMaker

### managed MLflow は使わない

移行表では MLflow server を SageMaker managed MLflow に対応させていたが、採用しない。

トラッキングサーバーは起動している限り課金が続く。Small サイズで us-east-1 が $0.642/時、フランクフルトが $0.886/時。月額にすると $500〜650 になる。torch を外してイメージを削り、表形式データを選んで学習コストを抑えた判断が、これ一つで帳消しになる。

AWS 側では SageMaker Model Registry の approval status を使う。MLflow の alias で設計してあるので読み替えは容易で、追加コストはゼロ。MLflow はローカル専用と割り切る。

### Serverless Inference では Model Monitor が使えない

Serverless Inference は、GPU、VPC 構成、マルチモデルエンドポイント、**データキャプチャ、Model Monitor** が機能除外になっている。

つまりドリフト監視は自前で組む必要がある。推論ログを MLflow に書かず JSON Lines を S3 に吐く設計にしたのは、結果としてこの制約下での唯一の解になっていた。`drift_check.py` は Lambda で実行する。

### 推論ログはバッファせず毎回書く

ローカルでは 10 件ずつバッファして S3 に書いているが、エンドポイントでは `INFERENCE_LOG_FLUSH_SIZE=1` を渡して毎回書かせる。

SageMaker のコンテナは停止時に `shutdown` が発火する保証がなく、バッファに残ったログは失われる。
ドリフト検知が推論ログに依存している以上、欠損は許容できない。

これは推測ではなく観測している。Batch Transform で `FLUSH_SIZE=1000000` を渡してバッファさせたところ、
ジョブは正常に完了したのにログは 1 件も書かれなかった。FastAPI の `lifespan` の終了処理
（`logger.flush()`）が呼ばれないまま、コンテナが止まっている。

なお Batch Transform では出力を SageMaker が S3 に書くので、自前のログ書き出しは
`INFERENCE_LOG_ENABLED=0` で止めている。

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

### Batch Transform で既存の推論イメージをそのまま使う

Batch Transform もエンドポイントと同じく `/ping` と `/invocations` を叩く。
エンドポイント用に作った推論イメージが、コード変更なしで動いた。

条件は Content-Type の指定。SageMaker で JSON Lines を扱う標準は `application/jsonlines` だが、
FastAPI は pydantic のモデルを受ける関数に `application/json` 以外が来ると 422 を返す。

    "ContentType": "application/json",
    "SplitType": "Line",
    "BatchStrategy": "SingleRecord",

`SplitType: Line` で入力ファイルを 1 行ずつ切り出し、各行を `application/json` として送る。
1 行 1 リクエストになるので大量データには向かないが、日次のコストデータなら十分。

### JoinSource で出力に特徴量を残す

Batch Transform の出力は、既定では予測結果だけになる。`drift_check` は特徴量の分布を比べるので、
予測だけでは使えない。

`DataProcessing.JoinSource: "Input"` を指定すると、入力の各行に予測結果が `SageMakerOutput` として結合される。
入力と出力を自前で突き合わせる層が要らなくなる。

出力にはどのモデルで推論したかが残らないので、推論 API のレスポンス自体に `model_version` を含めた。
`SageMakerOutput` の中に入る。

### 推論ログは 2 つの形式が混在する

`inference-logs/` には形式の違う 2 種類のファイルが並ぶ。

| 由来 | 特徴量 | 予測 |
|---|---|---|
| ローカル / エンドポイント（`inference_logger`） | `features` の下にネスト | `prediction` / `probability` |
| Batch Transform（JoinSource） | 最上位 | `SageMakerOutput` の下 |

`drift_check` はファイル単位で形式を揃えてから結合する。

最初は全ファイルを結合してから形式を判定していたが、混在すると `features` 列が一部の行だけ
NaN になり `json_normalize` が失敗する。日付で絞って読む分には問題が出ないので、
全期間を読んで初めて分かる種類の不具合だった。テストで固定している。

### SageMaker Model は Transform Job と同じライフサイクルで扱う

エンドポイントでは Model を Terraform で管理していたが、Batch Transform ではスクリプトが作って消す。
Transform Job が終われば Model は不要で、残すと実行のたびに溜まる（課金は無い）。

Step Functions に移すときも、同じワークフローの中で CreateModel → CreateTransformJob → DeleteModel と並べる。

### 現行モデルは「作成日時が最新の Approved」

MLflow の alias は 1 つのバージョンを指すが、SageMaker の approval status はパッケージごとの属性で、
複数の Approved が並びうる。そこで「作成日時が最新の Approved」を現行と定義した。

副次的に、ロールバックが「最新の Approved を Rejected にする」だけで済む。1 つ前の Approved が自動的に現行に戻る。

注意点として、作成日時で並べるので、古いパッケージを後から Approved にしても現行にはならない。
自動化したフローでは登録直後に判定するので順序は崩れないが、手で操作するときは意識が要る。

この定義は `src/registry.py` の `latest_approved()` とステートマシンの `ResolveModel` の 2 か所にある。
Step Functions の定義は Python から参照できないので、二重管理になっている。

### Processing Job は推論イメージで動かす

当初は学習イメージを流用するつもりだったが、学習イメージの依存は `pandas / pyarrow / numpy / scikit-learn`
の 4 つだけで、**boto3 が入っていない**。drift_check と promote は boto3 で S3 と SageMaker を叩くので動かない。

推論イメージには boto3 と `src/` 一式が入っているので、そちらを使っている。
FastAPI と uvicorn が余分に載るが数 MB。用途別に依存を分ける原則からは専用イメージを作るのが筋だが、
ECR リポジトリと lock と CI が 1 つずつ増えるので、今は流用にとどめている。

推論イメージは非 root で動くが、結果を S3 API で直接書き、ファイルシステムに書き込まないので問題にならなかった。

## パイプラインと失敗の扱い

### drift_check のクラッシュを「ドリフトあり」と区別する

`drift_check` の終了コードは 0（ドリフトなし）/ 1（ドリフトあり）/ 2（判定不能）で、
`retrain_pipeline.sh` は 1 のときに再学習へ進む。

ところが **Python の未捕捉例外も終了コード 1 を返す**。S3 の認証切れ、MLflow の停止、
想定外の形式のログ、いずれでもクラッシュすると「ドリフトを検知した」と解釈されて再学習が走っていた。
作業中だけでも SSO トークン切れと依存の欠落で 2 回クラッシュさせている。
AWS で自動化すると、これは無駄な Training Job の課金になる。

`promote.py` も同じ構造だが、あちらはクラッシュが 1（昇格拒否）になり、**安全側に倒れる**。
`drift_check` は**危険側に倒れる**点が違った。

対処は 2 か所。

- `drift_check` の想定外の例外を捕まえて終了コード 2 にする（意図した `SystemExit` は再送出する）
- `retrain_pipeline.sh` を「0 と 2 以外は再学習」から「1 だけが再学習、それ以外は判定不能」に反転する

存在しないバケットを指定して確かめた。

    drift_check failed: NoSuchBucket: An error occurred (NoSuchBucket) ...
    exit: 2

終了コードで分岐する設計は一般的だが、言語ランタイムが既定で返すコードと意味が衝突していないかは
確認しておく必要がある。

### 「現行モデルなし」を例外で判定しない

`promote.py` の MLflow 版は、現行モデルの取得を `except Exception` で囲み、
**全ての例外を「現行モデルなし」と解釈していた**。

MLflow サーバーへの接続失敗や認証エラーでも「初回昇格」扱いになり、現行と比較せずに下限チェックだけで
昇格する。drift_check のクラッシュが再学習を引き起こしていたのと同じ種類の問題で、
こちらは比較をすり抜けて本番モデルが入れ替わるので影響が大きい。

alias の有無を `get_registered_model(...).aliases` で明示的に確認し、それ以外の失敗は例外のまま止めるように直した。

### Processing Job では判定結果を終了コードで返さない

drift_check と promote は、ローカルでは終了コードで結果を返す（0: ドリフトなし / 1: ドリフトあり / 2: 判定不能）。
ところが Processing Job は 0 以外を全て**ジョブ失敗**として扱う。Step Functions から見ると
「ドリフトを検知した」と「クラッシュした」が同じ Failed になり、区別するには FailureReason の文字列を解析するしかない。

drift_check で直した「クラッシュとドリフトありが区別できない」問題が、別の層で再発する形だった。

`--result-s3-uri` を指定すると、判定結果を JSON で S3 に書き、判定できた場合は結果にかかわらず終了コード 0 を返す。
終了コードは「判定できたか」だけを表し、中身は JSON で受け渡す。
昇格拒否（Rejected）も正常な判定結果なので、ジョブは成功で終わる。

    {"drift": true, "drifted_features": ["cost", "cost_ma7", "cost_std7"], "samples": 200, "psi": {...}}
    {"model_package_arn": "...:model-package/cost-anomaly-detector/4", "approved": false}

### Step Functions の SageMaker 統合は AddTags を要求する

最初の実行は Transform の起動で即座に失敗した。

    not authorized to perform: sagemaker:AddTags on resource: ...transform-job/bt-...

Step Functions の `.sync` 統合は、起動したジョブに管理用のタグを自動で付ける。
パラメータに `Tags` を書いていなくても付与されるので、`sagemaker:AddTags` が要る。
ロールを設計するとき「Tags を渡していないので不要」と判断して外していた。
`createModel` の統合はタグを付けないので、Model の作成までは通っていた。

### Fail State で本当の原因を隠していた

上の AddTags の失敗は、最初 `describe-execution` では次のようにしか見えなかった。

    ["TransformFailed", "Batch Transform が失敗しました。Model は削除を試みています"]

Fail State に固定の文言を書いていたためで、本当の原因は実行履歴の `TaskFailed` イベントを掘らないと見えなかった。
Catch で `$.error` に保存していたのに、Fail State がそれを使っていなかった。

`ErrorPath` / `CausePath` で保存したエラーをそのまま出すように変えた。
どの段階で失敗したかは State 名で分かり、なぜ失敗したかは `cause` で分かる。

Serverless Inference の切り分けで「失敗したときに情報が出る作りになっているかで調査時間が桁で変わる」と
書いたのと同じことを、自分の設計でやっていた。

### Transform が失敗しても Model を消す

ステートマシンは Transform の前に SageMaker Model を作り、後で消す。
Transform が失敗したときも Catch から削除の State を通してから Fail で終わる。

AddTags の失敗はちょうどこの経路を通った。実行後に `list-models` が空だったので、
失敗経路での後片付けが実際に機能したことを確認できた。意図的に作りにくい経路なので、偶然とはいえ良い検証になった。

## 通知

### 失敗の通知はステートマシンの外で拾う

パイプラインの失敗は、Step Functions の実行状態の変化を EventBridge ルールで拾って SNS に送る。
ステートマシンの中に通知の State を書くと、ステートマシン自体が壊れたとき（定義の誤りで即座に落ちたときなど）に
通知も一緒に止まる。外から状態変化を見ていれば、どこで落ちても捕まえられる。

劣化と入れ替えの通知はステートマシンの中に置いている。判定結果は `promote.json` に入っていて、
既存の Choice の流れに State を足すだけで済むため。`register_and_promote.py` から SNS に直接送る方法もあったが、
スクリプトの責務は判定までにとどめ、通知は制御の流れの側に置いた。
「判定結果は JSON で返し、分岐はステートマシンが行う」という既存の分担と揃えるため。

### 通知の State に Catch を付けない

SNS への送信が失敗したら、実行はそのまま FAILED で終わる。Catch で握りつぶして Succeed に進めると、
「通知が届かなかった」ことが誰にも分からなくなる。FAILED で終われば、今度は EventBridge のルールが
失敗の通知を送る。

昇格の判定自体（Approved / Rejected）は Processing Job の中で確定しているので、通知が失敗しても結果は変わらない。

### SNS の件名は ASCII のみ

SNS の `Subject` は ASCII のみ、100 文字未満という制約がある。日本語を入れると Publish が失敗する。
件名は英語（`[ops-side-of-ml] model promoted` など）、本文は日本語にしている。

### SNS トピックは暗号化していない

AWS 管理キー（`alias/aws/sns`）で暗号化すると、EventBridge はそのキーを使う権限を持てないので送信に失敗する。
暗号化するにはカスタマー管理キーが要り、月 1 ドルかかる。通知の本文は実行名と判定の数値だけで機密ではないので、
暗号化しない判断をした。

トピックポリシーでは、EventBridge からの送信を失敗通知のルールに限定している（`aws:SourceArn`）。
Step Functions からの送信は同一アカウントなので、SFN ロールの IAM ポリシーだけで足りる。

### 購読は Terraform で管理しない

メールアドレスを公開リポジトリに書けないのと、メールの購読は受信者が確認リンクを押すまで保留になり、
Terraform ではその確認を完了できないため。トピックだけを Terraform で作り、購読は CLI で 1 回だけ登録する。
購読は環境ごとに変わる値で、コードとして管理する利点が薄い。

### 失敗通知は安く確かめられる

`input_prefix` に S3 の URI でない文字列を渡すと、Transform の起動時に SageMaker が即座に拒否し、数秒で FAILED になる。
その前の CreateModel は無料で、失敗経路の後片付けで削除される。

    aws stepfunctions start-execution --state-machine-arn <arn> \
      --input '{"input_prefix":"invalid","training_input_prefix":"invalid","eval_prefix":"invalid"}'

最初は `eval_prefix` を省けば安く失敗させられると考えたが、そのエラーは RegisterAndPromote に入った時点で起きる。
その前に Transform・DriftCheck・Train が全て走るので、10 分かかり課金も発生する。失敗させる位置を選ぶ必要がある。

ルールの対象はステートマシンの ARN を文字列で組み立てて指定している。ステートマシンの属性を参照すると、
定義を変えるたびに plan でルールも変更扱いになるため（scheduler のポリシーと同じ理由）。

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
