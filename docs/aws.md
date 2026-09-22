# AWS で動かす

ローカルと同じコードを SageMaker で動かし、Step Functions でループを自動化した。

## 構成

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
      ├─ Processing Job ────── register_and_promote ── 現行と候補を同じホールドアウトで評価
      │                                                  → Approved / Rejected
      └─ SNS ───────────────── 現行の劣化 / モデルの入れ替えを通知

    EventBridge ルール ─── 実行が FAILED / TIMED_OUT / ABORTED → SNS → メール

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

### 通知

| 事象 | 経路 | 動作確認 |
|---|---|---|
| パイプラインの失敗（FAILED / TIMED_OUT / ABORTED） | EventBridge ルール → SNS | 実地で確認（メール受信） |
| 現行モデルがホールドアウトで下限を割った | ステートマシン → SNS | 定義のみ |
| モデルが入れ替わった | ステートマシン → SNS | 定義のみ |

「ドリフトを検知したが候補は拒否された」は通知しない。現行が下限を割っていなければ、対処が要らないため。

劣化と入れ替えの通知は、実際に起きるまで動作が保証されていない。現行の v6 はホールドアウトで下限を満たしており、
入れ替えには McNemar を満たす改善が要るので、意図的に起こすのが難しい。

購読は Terraform の外で登録している。

    aws sns subscribe --topic-arn <alerts_topic_arn> --protocol email --notification-endpoint <address>

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
