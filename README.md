# ops-side-of-ml

MLflow + MinIO によるローカル MLOps 練習環境。インフラエンジニアが ML のワークフローを手を動かして学ぶためのリポジトリ。

## 構成

| サービス | 役割 | ポート |
|---|---|---|
| mlflow-server | 実験管理・Model Registry | 5001 |
| s3 (MinIO) | S3互換ストレージ（artifact保存） | 9005 / 9006 |
| ml-app | 学習スクリプト実行・推論API | 8000 |

## セットアップ

```bash
cp .env.example .env
# .env を編集。MINIO_ROOT_PASSWORD にランダムな値を設定する
#   例: openssl rand -hex 32

docker compose up -d
```

## 動作確認

```bash
docker compose exec ml-app python train.py
```

- MLflow UI: http://localhost:5001
- MinIO Console: http://localhost:9006

## 今後の予定

- [ ] Model Registry を使った学習と推論の接続
- [ ] 推論ログの分離とドリフト検知
- [ ] AWS (SageMaker) への移行