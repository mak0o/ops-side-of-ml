#!/usr/bin/env bash
# ドリフト検知 → 再学習 → 昇格判定 を通しで実行する。
#
# 終了コード:
#   0 = 完了（ドリフトなし、または再学習して昇格した）
#   1 = 再学習したが昇格基準を満たさなかった（要確認）
#   2 = 判定不能（推論ログがない、メトリクス欠損など）

set -uo pipefail

RUN="docker compose run --rm ml-app"
DRY_RUN="${DRY_RUN:-0}"

echo "=== 1. ドリフト検知 ==="
$RUN python -m src.drift_check
drift_status=$?

case $drift_status in
  0)
    echo
    echo "ドリフトなし。再学習は不要です。"
    exit 0
    ;;
  2)
    echo
    echo "判定できませんでした。推論ログを確認してください。"
    exit 2
    ;;
esac

echo
echo "=== 2. 再学習 ==="
$RUN python -m src.train --register || exit 2

echo
echo "=== 3. 最新バージョンの取得 ==="
latest=$($RUN python -c "
import mlflow
c = mlflow.MlflowClient()
versions = c.search_model_versions(\"name='cost-anomaly-detector'\")
print(max(int(v.version) for v in versions))
" | tr -d '\r')

if ! [[ "$latest" =~ ^[0-9]+$ ]]; then
  echo "バージョンの取得に失敗しました: '$latest'"
  exit 2
fi
echo "candidate: version $latest"

echo
echo "=== 4. 昇格判定 ==="
if [ "$DRY_RUN" = "1" ]; then
  $RUN python -m src.promote --candidate "$latest" --dry-run
  exit $?
fi

$RUN python -m src.promote --candidate "$latest"
promote_status=$?

if [ $promote_status -eq 0 ]; then
  echo
  echo "=== 5. 推論APIの再起動 ==="
  docker compose restart ml-app
  echo "完了しました。"
fi

exit $promote_status