FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements-serve.lock .
RUN pip install --no-cache-dir -r requirements-serve.lock

COPY src/ ./src/

# SageMaker は `docker run <image> serve` で起動する。
# PATH 上に serve という実行ファイルが必要。
COPY docker/serve-entrypoint.sh /usr/local/bin/serve
RUN chmod +x /usr/local/bin/serve

# 推論エンドポイントは常時リクエストを受けるので非 root で動かす。
# 権限変更が必要な処理を全て終えてから切り替える。
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8080