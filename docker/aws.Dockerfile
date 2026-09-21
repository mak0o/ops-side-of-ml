FROM python:3.12-slim
RUN pip install --no-cache-dir boto3 pandas pyarrow
WORKDIR /work