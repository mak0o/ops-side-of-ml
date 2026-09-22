## 区切り

2026 年 9 月時点で、当初の目標（ローカルで組んだ MLOps のループを AWS のマネージドサービスで組み直す）を
達成したものとして一区切りにしている。

### 対応したこと

- [x] Terraform で S3 と IAM を構築、state を S3 バックエンドへ
- [x] GitHub Actions から OIDC で plan / apply / push
- [x] ECR と SageMaker Training Job で学習を AWS に出す
- [x] エンドポイントを立てて推論ログを S3 に溜める（Real-time で確認）
- [x] Batch Transform でバッチ推論し、特徴量込みの推論ログを S3 に出す
- [x] drift_check を AWS 上のデータとベースラインだけで動かす
- [x] Model Registry の approval status で昇格を管理する
- [x] Batch Transform → drift_check → 再学習 → 昇格判定 を Step Functions で自動化
- [x] 昇格判定を同じホールドアウトでの比較にする（現行モデルの劣化も測る）
- [x] 昇格に差が偶然でないかの条件を設ける（片側 McNemar、α = 0.05）
- [x] パイプラインの失敗、現行の劣化、モデルの入れ替えを通知する

### 区切り時点で対応しなかったもの

いずれも意図的に見送った。

| 項目 | 見送った理由 |
|---|---|
| `latest/` にデータを置く上流 | 合成データのために偽物の上流を作り込むことになる。本物のデータ（自分の AWS アカウントの Cost and Usage Report など）で運用する段階で必要になる |
| 日次スケジュールの有効化 | 上流が無いので、有効にすると毎日データ無しで失敗する。スケジュール自体は Terraform で作成済みで、無効にしてある |
| Processing Job 用の専用イメージ | 推論イメージの流用で機能上の問題は無い。依存を用途別に分ける原則からの逸脱として記録している |
| Serverless Inference の解決 | Batch Transform に切り替え済みで、解いても構成は変わらない。切り分けの記録は [serverless.md](serverless.md) |
| 判定閾値の最適化、学習ウィンドウの自動決定 | ML 側の深掘りで、運用側から実装するという趣旨の外 |