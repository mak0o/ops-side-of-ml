FROM python:3.12-slim

# SageMaker が見る規約。バッファリングを切らないとログが遅延する。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 依存は lock で固定。mlflow はローカル専用なので含めない。
COPY requirements-train.lock .
RUN pip install --no-cache-dir -r requirements-train.lock

COPY src/ ./src/

# SageMaker は `docker run <image> train` で起動する。
# PATH 上に train という実行ファイルが必要。
COPY docker/train-entrypoint.sh /usr/local/bin/train
RUN chmod +x /usr/local/bin/train