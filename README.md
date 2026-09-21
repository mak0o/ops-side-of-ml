# ops-side-of-ml

インフラエンジニアが MLOps を運用側から実装した練習リポジトリ。
題材は **AWS コストの異常検知**。合成データを使い、学習からドリフト検知・再学習判定までのループをローカルで一周させている。

同じループを AWS (SageMaker + Step Functions) に載せた。バッチ推論 → ドリフト検知 → 再学習 → 昇格判定が
人手を介さずに一周し、ローカルと同じ検知結果が再現できている。

当初は昇格判定で**別々のデータで測った数値を比べていた**。期間で切ったホールドアウトで現行と候補を
同じ条件で評価するように直し、その結果、過去に導いた結論の一つが比較方法の欠陥による見かけだったと分かった
（[昇格判定を同じデータで比べるようにした](#昇格判定を同じデータで比べるようにした)）。

残っている大きな論点は、**昇格が差の大きさを問わない**こと。初めての自動入れ替えは、300 行のうち 3 行の違いで決まった。

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

    Step Functions（EventBridge Scheduler から日次。現在は無効）
      │
      ├─ ResolveModel ──────── Model Registry の最新 Approved を解決
      ├─ Batch Transform ───── 推論ログ (S3, 特徴量 + 予測)
      ├─ Processing Job ────── drift_check ── baseline.json (model.tar.gz から)
      │      └─ ドリフトなし → 終了
      ├─ Training Job ──────── model.tar.gz (S3)
      └─ Processing Job ────── register_and_promote ── 現行と候補を同じホールドアウトで評価
                                                         → Approved / Rejected

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
> 経緯は [過去の結論の見直し](#過去の結論の見直し) を参照。当初の記述はそのまま残している。

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

## AWS でバッチ推論する

SageMaker Batch Transform で、S3 のデータをまとめて推論する。

    docker compose --profile tools run --rm awscli python scripts/batch_transform.py \
      --data data/cost.parquet \
      --model-artifact s3://.../training-output/<job>/output/model.tar.gz \
      --image-tag <commit sha> \
      --limit 200 --sample --join-input --wait

スクリプトは次の順で動く。

1. parquet を JSON Lines に変換して S3 に置く
2. SageMaker Model を作る（推論イメージ + `model.tar.gz`）
3. Transform Job を実行して完了を待つ
4. Model を削除する

出力は推論ログと同じ日付パーティションに書く。

    s3://<bucket>/inference-logs/year=2026/month=09/day=21/<job>/input.jsonl.out

1 行に特徴量と予測結果が並ぶ。予測は `SageMakerOutput` の下に入る。

    {"SageMakerOutput":{"is_anomaly":0,"model_version":"1835041f...","probability":0.0347},
     "cost":28.6,"cost_ma7":26.56,"cost_ratio_ma7":1.08,"cost_std7":1.80,
     "cost_vs_lastweek":0.98,"day_of_week":2,"is_weekend":0}

`ml.m5.large` で 1 回あたり数分、課金秒数は 1 秒未満（`BillableTimeInSeconds: 0`）。
待ち時間の大半はインスタンスの起動。

## AWS でドリフト検知する

`model.tar.gz` に同梱した `baseline.json` と、S3 の推論ログを比較する。MLflow にも MinIO にも依存しない。

    docker compose --profile tools run --rm awscli sh -c \
      "INFERENCE_LOG_BUCKET=ops-side-of-ml-<account> python -m src.drift_check \
        --model-artifact s3://.../training-output/<job>/output/model.tar.gz \
        --prefix inference-logs/year=2026/month=09/day=21/<job>/"

`--model-artifact` を渡すと S3 の tar から、渡さなければ従来どおり MLflow からベースラインを読む。

### ローカルと同じ結果が再現できるか

Batch Transform にランダム抽出した 200 件を流して確かめた。

**通常データ（学習と同じ母集団）**

    feature                   PSI  status        base_mean  curr_mean
    ------------------------------------------------------------------
    cost                   0.0381  stable            86.96      82.76
    cost_ma7               0.0266  stable            86.32      82.51
    cost_std7              0.0392  stable            27.19      27.18
    cost_ratio_ma7         0.0166  stable             1.01       1.02
    cost_vs_lastweek       0.0518  stable             1.11       1.06

    no significant drift

**コスト水準 1.5 倍のデータ（`--drift 1.5`）**

    feature                   PSI  status        base_mean  curr_mean
    ------------------------------------------------------------------
    cost                   0.9864  significant       86.96     114.74
    cost_ma7               4.2987  significant       86.32     114.55
    cost_std7              0.5120  significant       27.19      31.36
    cost_ratio_ma7         0.0848  stable             1.01       1.00
    cost_vs_lastweek       0.0625  stable             1.11       1.05

    DRIFT DETECTED: cost, cost_ma7, cost_std7

絶対値の特徴量が反応し、比率の特徴量は stable のまま。ローカルで確認した
「利用規模が拡大しただけで、異常の出方そのものは変わっていない」という読み方が AWS 上でも成立する。

PSI の値そのものはローカルの結果と一致しない。ベースラインが別のモデル（学習データの範囲が違う）で、
抽出方法も違う（ローカルは先頭 400 件、こちらはランダム 200 件）ため。見るべきは傾向。

`cost_ma7` の PSI が突出しているのは、7 日移動平均が平滑化されていて分布の幅が狭いから。
同じ水準のずれでも、幅の狭い分布では大半の値がベースラインのビンの外に出る。

### 最初の検証は間違った結論を出していた

最初はデータの先頭 50 件（`head(50)`）で試し、次の結果になった。

    cost                   7.1212  significant       86.96      31.01
    cost_ratio_ma7         1.4426  significant        1.01       1.02
    cost_vs_lastweek       0.3468  significant        1.11       1.10

    DRIFT DETECTED: cost, cost_ma7, cost_std7, cost_ratio_ma7, cost_vs_lastweek

原因は 2 つ重なっていた。

- 先頭 50 件は合成データの初期で、利用規模が小さい期間に偏っている（`cost` の平均が 31）
- 件数が少なすぎて PSI が不安定になっている

後者は、比率の特徴量が **平均はほぼ同じ（1.01 と 1.02）なのに significant** になっていることで分かる。
PSI は 10 ビンに分けて構成比を比べるので、50 件だと 1 ビンあたり 5 件程度しかない。
空のビンや 1 件だけのビンができると値が跳ねる。ランダム 200 件にすると同じ特徴量が 0.017 に落ちた。

そのため `drift_check` に件数の下限を設け、下回ったら判定不能（終了コード 2）で止めるようにした。

    samples 50 < 100: 判定に必要な件数に達していません

100 は「10 ビンで各ビンに 10 件程度」という目安で、厳密な根拠はない。`--min-samples` で変えられる。

## AWS でモデルを登録・昇格する

ローカルの MLflow Registered Model に相当するものとして、SageMaker Model Registry の
Model Package Group `cost-anomaly-detector` を使う。

| | ローカル（MLflow） | AWS（SageMaker） |
|---|---|---|
| 登録 | `train.py --register` | `train_sagemaker.py --register` |
| メトリクス | run の metrics | Model Package の `CustomerMetadataProperties` |
| 現行モデル | `@production` alias | 作成日時が最新の `Approved` |
| 昇格 | alias を付け替え | status を `Approved` に |
| 拒否 | 何もしない | status を `Rejected` にし、理由を記録 |

判定ロジック（`evaluate()`）はレジストリに依存しない純粋関数にしてあったので、そのまま流用している。
差し替えたのは「メトリクスの取得」と「昇格の書き込み」だけ。

    docker compose --profile tools run --rm awscli python scripts/train_sagemaker.py \
      --wait --register --serve-image-tag <commit sha>

    docker compose --profile tools run --rm awscli python -m src.promote \
      --model-package-arn arn:aws:sagemaker:...:model-package/cost-anomaly-detector/1

`batch_transform.py` と `drift_check.py` は `--model-package-group` を渡すと、最新の Approved から
モデル・推論イメージ・ベースラインを自分で解決する。パッケージがイメージと成果物をセットで持っているので、
グループ名だけで全てが決まる。推論結果の `model_version` にも `cost-anomaly-detector/1` のように
モデルの世代が残る。

### 初回昇格は無条件ではない

v1 を昇格させる前に、わざと弱いモデル（`n_estimators=1, max_depth=1`、つまり切り株）を v2 として判定した。

    current: none (初回昇格)
    candidate(v2): f2=0.688  precision=0.398  recall=0.841

    baseline     現行モデルなし、初回昇格          PASS
    precision    0.398 >= 0.60                  FAIL
    recall       0.841 >= 0.80                  PASS

    REJECT: v2 は昇格基準を満たしません

比較対象が無い初回は F2 の比較が効かないので、下限チェックだけが防御になる。
F2 と別に下限を設けた理由がここにある。

v1 と v2 の recall は **0.8409 で完全に同じ**だった。切り株でも `class_weight="balanced"` のおかげで
異常をほぼ同じだけ拾えるが、代わりに正常を大量に異常と誤判定して precision が崩壊している。
「recall さえ上がれば F2 は改善しうる」という懸念の具体例になっている。

### 判定の理由をパッケージに残す

拒否したパッケージには理由が記録される。コードもログも見ずに、SageMaker の状態だけから追える。

    $ aws sagemaker describe-model-package --model-package-name <arn> \
        --query '[ModelApprovalStatus,ApprovalDescription]'
    ["Rejected", "promote.py: precision FAIL (0.398 >= 0.60)"]

最初は `precision: 0.398 >= 0.60` と判定条件だけを書いていたが、数式として偽の文字列が
「条件を満たした」ようにも読めるので、`FAIL` / `PASS` を明示する形に変えた。
監査記録は、後から読む人が文脈を持っていない前提で書く必要がある。

## AWS でループを回す

Step Functions で、バッチ推論から昇格判定までを 1 つのワークフローにした。

    aws stepfunctions start-execution \
      --state-machine-arn arn:aws:states:...:stateMachine:ops-side-of-ml-pipeline \
      --input '{"input_prefix": "s3://.../batch-input/timeline/",
                "training_input_prefix": "s3://.../training-input/timeline/",
                "eval_prefix": "s3://.../eval-input/timeline/"}'

推論データ、再学習データ、ホールドアウトの場所は実行時の入力で受け取る。データを S3 に置く役は
ワークフローの外にある（本来は Cost and Usage Report のエクスポートとラベル付けの仕組みが担う位置）。
`eval_prefix` が無いと RegisterAndPromote のパラメータを組み立てられずに実行が失敗する。
登録時の指標で比べる古い経路に黙って戻らないよう、ホールドアウトを必須にしている。

Batch Transform と Training Job は Step Functions の SageMaker 統合（`.sync`）で直接呼び、
drift_check と登録・昇格判定は Processing Job で動かす。

### 2 回の実行結果

**通常データ（学習と同じ母集団からランダム 200 件）**

    Prepare → BuildPaths → ResolveModel → HasApprovedModel → DescribeModel → CreateModel
      → Transform → DeleteModel → DriftCheck → ReadDriftResult → IsDrift → NoDrift

drift.json の PSI は手動で実行したときと小数点以下 4 桁まで一致した（cost 0.0381、cost_ma7 0.0266 …）。
同じ 200 件、同じモデル、同じベースラインから同じ値が出ているので、パイプラインの各段で情報が落ちていない。

**コスト水準 1.5 倍のデータ**

    ... → IsDrift → Train → RegisterAndPromote → ReadPromoteResult → IsApproved → Rejected

    current  (v1): f2=0.815  precision=0.726  recall=0.841
    candidate(v4): f2=0.760  precision=0.963  recall=0.722

    f2           0.815 -> 0.760 (-0.055)        FAIL
    precision    0.963 >= 0.60                  PASS
    recall       0.722 >= 0.80                  FAIL

ドリフトを検知し、再学習し、登録し、判定して拒否した。v1 が現行のまま残っている。

v4 は v2 と逆の崩れ方をしている。誤報はほぼ無いが、異常を 4 件に 1 件以上見逃す。
ローカルで「直近のみで学習すると recall が落ちる」（precision 0.889 / recall 0.667）と観察した現象が、
AWS 上の自動パイプラインでも再現した。

所要時間は通常経路で約 8 分、再学習まで進むと約 10 分。大半はジョブごとのインスタンス起動。費用は 1 回あたり数円。

## 昇格判定を同じデータで比べるようにした

### 当初の欠陥

Step Functions で v4 を判定したとき、比べた 2 つの F2 は別々のデータで測った値だった。

- v1 の F2 は、元のデータを学習用とテスト用に分けたときの**元データのテスト部分**での成績
- v4 の F2 は、ドリフト後のデータを分けたときの**ドリフト後データのテスト部分**での成績

別々の試験問題で取った点数を並べて、高いほうを採用していた。これは AWS に移して生じた問題ではなく、
**ローカルの `promote.py` の設計から持ち込んでいた**。登録時に保存した指標で比べる方式は、再評価の手間を省いた近道だった。

v4 の拒否は、F2 の比較ではなく recall の下限（v4 自身のテストデータで 0.722 < 0.80）で成立していた。
下限を比較と独立に設けていたことが、意図しない形で効いた。

加えて、学習時の評価そのものも楽観的だった。`train.py` はランダムに 8:2 で分けているが、特徴量に
7 日移動平均が入っているので、隣り合う日が学習側とテスト側に分かれると、移動平均を通じて情報が漏れる。

### 期間で切ったホールドアウト

時系列として連続したデータを作り、最後の 60 日をホールドアウトにした。

    docker compose run --rm --no-deps ml-app python -m src.generate_data \
      --append-to data/cost.parquet --days 180 --drift 1.5 --seed 99 \
      --out data/cost_timeline.parquet

    docker compose --profile tools run --rm awscli python scripts/prepare_datasets.py \
      --data data/cost_timeline.parquet --name timeline

    train:   2025-01-22 .. 2026-08-28  2920 rows, 259 anomalies
    holdout: 2026-08-29 .. 2026-10-27  300 rows, 30 anomalies

2025 年の元の水準に、水準 1.5 倍の 180 日を続けた 1 本の時系列。直近 60 日は 2 つに分けて置く。

- 特徴量のみ → 推論に流す（本番で観測するデータ）
- 特徴量 + ラベル → ホールドアウト（後から判明した正解）

本番で言えば、推論した期間についてアラートの真偽が判明し、ラベルが付いた状態にあたる。
ホールドアウトは学習データと期間が重ならないので漏れがない。

昇格判定では、現行モデルと候補モデルの `model.tar.gz` を読み、両方をホールドアウトで推論して、
**同じ関数（`src/metrics.py`）**で指標を計算して比べる。学習時の評価も同じ関数を使うので、
定義が 2 か所に分かれて比較が壊れることがない。判定ロジック（`evaluate()`）は一行も変えていない。

ホールドアウトの異常が 10 件未満なら判定不能で止める。F2 は正例の件数に強く依存するので、
少なすぎると数件の差で結果が振れる。

### v1 はドリフト後も下限を割っていなかった

2025 年の水準で学習した v1 を、水準 1.5 倍のホールドアウトで評価した。

| | 学習時（ランダム分割） | ホールドアウト |
|---|---|---|
| F2 | 0.815 | 0.877 |
| precision | 0.726 | 0.794 |
| recall | 0.841 | 0.900 |

Step Functions の 2 回目の実行では、drift_check が「ドリフトあり」と判定して再学習を走らせていた。
しかし現行モデルは、少なくとも下限を割るほどには劣化していなかった。

v1 がどの特徴量に頼っているかを見ると、理由が分かる。

    cost_vs_lastweek     0.457   stable
    cost_ratio_ma7       0.210   stable
    cost                 0.124   significant
    cost_ma7             0.089   significant
    cost_std7            0.088   significant
    day_of_week          0.022   stable
    is_weekend           0.009   stable

判断の約 3 分の 2 は比率の特徴量 2 つに依存していて、そのどちらも安定していた。drift_check が反応した
3 つの特徴量は、合計しても重要度の 3 割程度。**入力の分布が変わったことと、モデルの性能が落ちたことは別の事象**で、
drift_check は前者しか見ていない。

ただしこの重要度はランダムフォレストの不純度ベースのもので、連続値の特徴量を過大に評価する傾向がある。
また、異常 30 件で測った 0.815 と 0.877 の差に意味があるとは言えない。言えるのは「下限は割っていない」まで。

### 初めての自動入れ替え

timeline データでパイプラインを回した。ドリフトを検知し、旧データ + 新しい水準の最初の 120 日で再学習し、
v1 と同じホールドアウトで比べた。

    holdout: 300 rows, 30 anomalies
    current  (v1): f2=0.877  precision=0.794  recall=0.900
    candidate(v6): f2=0.894  precision=0.871  recall=0.900

    f2           0.877 -> 0.894 (+0.017)        PASS
    precision    0.871 >= 0.60                  PASS
    recall       0.900 >= 0.80                  PASS

    PROMOTED: v6 -> Approved

パイプラインが初めて人手を介さずにモデルを入れ替えた。

recall は同じで、precision だけが上がっている。異常 30 件に対してどちらも 27 件を検出し、
違いは誤報の数（逆算すると v1 が 7 件、v6 が 4 件）。v1 の誤報の一部は、コスト水準が上がったことで
正常な日を異常と読んでいた可能性がある。重要度の 3 割を占める絶対値の特徴量が、precision の側に効いていたという解釈。

「v1 は劣化していなかった」は、下限を割っていないという意味では正しいが、改善の余地が無かったという意味では誤りだった。

### 過去の結論の見直し

v6 は旧データと新データを混ぜて学習したモデルで、同じホールドアウトで比べたら v1 より良かった。

[再学習は必ずしも改善しない](#再学習は必ずしも改善しない) では「混在させると 2 つの分布の中間を学んでしまう」と
結論していた。この結論は別々のデータで測った数値を並べて導いたもので、比較方法の欠陥による見かけだった可能性が高い。

「直近のみで学習すると recall が落ちる」も同じ比較方法で得た観察だが、v4（ドリフト後のデータのみで学習）が
recall の下限を割った件は v4 自身のテストデータでの絶対値なので、こちらはある程度は信頼できる。

### 「劣化していなければ再学習しない」は入れなかった

v1 が下限を割っていないと分かった時点で、次の分岐を足すことを検討した。

    DriftCheck → ドリフトあり → 現行を評価 ─┬─ 下限を割っている → 再学習
                                            └─ 割っていない     → 終了

不要な再学習を省ける構成に見えるが、これを入れていたら **v6 は作られなかった**。v1 は下限を満たしているので
再学習が走らず、誤報を減らす改善を取りこぼしていた。

再学習は 1 回数円で、判定は公平なホールドアウトで行う。空振りの再学習のコストより、改善を取りこぼすコストのほうが大きい。
ドリフトを検知したら再学習し、同じ条件で比べて判定する、という今の構成を維持している。

現行モデルがホールドアウトで下限を割っていた場合は、結果に `current_degraded: true` を記録する。
候補が拒否され、かつ現行も劣化している状態を黙って放置しないための足がかりで、通知はまだ無い。

### 3 行の差で昇格が決まった

v1 と v6 の違いは、300 行のうち誤報 3 行だった。今の判定は `候補の F2 > 現行の F2` で、差の大きさを問わない。

異常 30 件、正常 270 件で測った F2 の 0.017 差は、データの揺らぎで十分起こりうる大きさ。今回は v6 が
良いと考える理由（新しい水準を見ている）があるが、判定のロジック自体はその理由を知らない。
わずかな差で入れ替わり続けることを許すと、推論結果の一貫性が損なわれる。未解決。

### ラベルがある前提に立っている

この仕組みは、直近期間の正解ラベルが揃っていることを前提にしている。合成データでは即座に生成できるが、
実運用ではラベルは遅れて届き、[推論ログからラベルを得る手段がない](#推論ログからラベルを得る手段がない) に直結する。

入力のドリフトはラベル無しで即座に分かるが、今回のように性能と一致しないことがある。
性能はラベルが揃うまで分からない。この 2 つをどう組み合わせるかは、このリポジトリでは解いていない。

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

### その後の判断

Serverless の解決は保留し、推論は Batch Transform に切り替えた。

題材は日次のコストデータで、リアルタイム性の要求がない。常時アクセスできる推論 API は
ローカルで FastAPI を使った延長で選んでいただけで、ユースケースからは導かれていなかった。
Batch Transform なら実行時だけ課金され、Serverless の問題も迂回できる。

推論イメージは作り直していない。エンドポイント用に作った `/ping` と `/invocations` を
Batch Transform もそのまま使う（詳細は AWS 側の設計判断を参照）。

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
| `infra/` | Terraform。S3 / IAM / ECR / SageMaker Model Registry |
| `infra/pipeline.tf` | Step Functions のステートマシン、そのロール、日次スケジュール（無効） |
| `infra/bootstrap/` | state バケットを作る。ここだけローカル state |
| `docker/` | 用途別の Dockerfile（ml-app / mlflow / train / serve / terraform / aws） |

## テスト

    docker compose run --rm --no-deps ml-app sh -c \
      "pip install -q -r requirements-dev.lock && python -m ruff check . && python -m pytest -q"

PSI 計算、昇格判定ロジック、推論ログの形式の正規化、Model Registry からの現行モデルの解決と登録、
ホールドアウトの検査と評価（学習時の列順で推論すること）、指標の計算をカバーしている。
CI でも lint とあわせて実行される。

Model Registry のテストでは SageMaker の偽物を使い、`list_model_packages` に渡す並び順の引数
（`SortBy="CreationTime"`, `SortOrder="Descending"`）を偽物の側で検査している。
「最新の Approved が現行」という定義はこの引数に依存しているので、誰かが変えたらテストで気づける。

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
PSI は件数が多いときにこの過敏さが出にくく、0.1 / 0.25 という実務的な閾値が確立している。

ただし逆に、**件数が少ないと PSI は不安定になる**。同じ母集団から取った 50 件で significant が出た
（[最初の検証は間違った結論を出していた](#最初の検証は間違った結論を出していた)）。
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
月 100 円前後になる。スケジュールは Terraform で作ってあるが、昇格判定の欠陥が直るまで無効にしている。

## AWS 側の設計判断

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

### 現行モデルは「作成日時が最新の Approved」

MLflow の alias は 1 つのバージョンを指すが、SageMaker の approval status はパッケージごとの属性で、
複数の Approved が並びうる。そこで「作成日時が最新の Approved」を現行と定義した。

副次的に、ロールバックが「最新の Approved を Rejected にする」だけで済む。1 つ前の Approved が自動的に現行に戻る。

注意点として、作成日時で並べるので、古いパッケージを後から Approved にしても現行にはならない。
自動化したフローでは登録直後に判定するので順序は崩れないが、手で操作するときは意識が要る。

この定義は `src/registry.py` の `latest_approved()` とステートマシンの `ResolveModel` の 2 か所にある。
Step Functions の定義は Python から参照できないので、二重管理になっている。

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

### Processing Job は推論イメージで動かす

当初は学習イメージを流用するつもりだったが、学習イメージの依存は `pandas / pyarrow / numpy / scikit-learn`
の 4 つだけで、**boto3 が入っていない**。drift_check と promote は boto3 で S3 と SageMaker を叩くので動かない。

推論イメージには boto3 と `src/` 一式が入っているので、そちらを使っている。
FastAPI と uvicorn が余分に載るが数 MB。用途別に依存を分ける原則からは専用イメージを作るのが筋だが、
ECR リポジトリと lock と CI が 1 つずつ増えるので、今は流用にとどめている。

推論イメージは非 root で動くが、結果を S3 API で直接書き、ファイルシステムに書き込まないので問題にならなかった。

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

### 権限の追加とリソースの作成を別の PR にする

apply ロールに `states:*` と `scheduler:*` を足す変更と、ステートマシンを作る変更は PR を分けた。

同じ PR にすると、Terraform は依存関係の無い両者を並列に作ろうとし、権限が反映される前に
ステートマシンの作成が走って失敗する。CI で apply するロールの権限を広げるときは、先にそれだけを適用する。

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
| Model Registry alias | SageMaker Model Registry approval status | 完了 |
| `train.py` | SageMaker Training Job | 完了 |
| FastAPI (Docker) | SageMaker Batch Transform | 完了（Serverless Inference は**失敗**して保留） |
| `drift_check.py` | Processing Job | 完了 |
| `promote.py` | Processing Job（登録と一体） | 完了 |
| `retrain_pipeline.sh` | Step Functions + EventBridge Scheduler | 完了（スケジュールは無効） |

## 今後

AWS

- [x] Terraform で S3 と IAM を構築、state を S3 バックエンドへ
- [x] GitHub Actions から OIDC で plan / apply / push
- [x] ECR と SageMaker Training Job で学習を AWS に出す
- [x] エンドポイントを立てて推論ログを S3 に溜める（Real-time で確認）
- [x] Batch Transform でバッチ推論し、特徴量込みの推論ログを S3 に出す
- [x] drift_check を AWS 上のデータとベースラインだけで動かす
- [x] Model Registry の approval status で昇格を管理する
- [x] Batch Transform → drift_check → 再学習 → 昇格判定 を Step Functions で自動化
- [x] 昇格判定を同じホールドアウトでの比較にする（現行モデルの劣化も測る）
- [ ] 昇格に最小改善幅を設ける（わずかな差で入れ替わらないように）
- [ ] `latest/` にデータを置く上流を用意する
- [ ] 日次スケジュールを有効化する（上流が用意できてから）
- [ ] 現行モデルが下限を割ったときに通知する
- [ ] Processing Job 用の専用イメージを作る
- [ ] Serverless Inference での作成失敗を解決する（保留）

その他

- [ ] 判定閾値の最適化とモデルへの記録
- [ ] 学習ウィンドウの自動決定

## 未解決の課題

### 昇格が差の大きさを問わない

詳細は [3 行の差で昇格が決まった](#3-行の差で昇格が決まった)。

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

### 学習データとモデルバージョンの対応（解決済み）

以前は `MODEL_VERSION` に推論イメージのコミットハッシュを入れていた。
Model Registry の導入後は、推論結果に `cost-anomaly-detector/1` のように Model Package のバージョンが入る。

ただし「そのモデルをどのデータで学習したか」は、Model Package のメタデータの `training_job` から
学習ジョブを辿り、その入力の S3 パスを見る必要がある。直接は記録していない。

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