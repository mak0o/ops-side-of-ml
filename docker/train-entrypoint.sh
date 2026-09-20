#!/bin/sh
# SageMaker から `train` として呼ばれる。
set -e
cd /app
exec python -m src.train