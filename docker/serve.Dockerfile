FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements-serve.lock .
RUN pip install --no-cache-dir -r requirements-serve.lock

COPY src/ ./src/

# 推論エンドポイントは常時リクエストを受けるので非 root で動かす。
# /opt/ml/model は SageMaker が読み取り専用でマウントするため書き込みは不要。
RUN useradd --create-home --uid 1000 appuser
USER appuser

# SageMaker は `docker run <image> serve` で起動する。
COPY docker/serve-entrypoint.sh /usr/local/bin/serve
EXPOSE 8080