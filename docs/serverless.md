# Serverless Inference の失敗記録

原因を特定できなかった失敗の記録。潰した仮説と、切り分けの過程を残している。

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
