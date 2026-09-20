#!/bin/sh
# SageMaker から `serve` として呼ばれる。
# ポートは 8080 固定（SageMaker の規約）。
set -e
cd /app
exec uvicorn src.serve:app --host 0.0.0.0 --port 8080 --workers 1