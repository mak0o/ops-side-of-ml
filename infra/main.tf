# アカウントIDを取得（バケット名の一意性確保に使う）
data "aws_caller_identity" "current" {}

locals {
  bucket_name = "${var.project}-${data.aws_caller_identity.current.account_id}"
}

# MLflow artifact と推論ログの保存先
resource "aws_s3_bucket" "artifacts" {
  bucket = local.bucket_name
}

# 誤ってパブリック公開しないための多重防御
resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# 推論ログは増え続けるので、古いものは自動削除する
resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    id     = "expire-inference-logs"
    status = "Enabled"

    filter {
      prefix = "inference-logs/"
    }

    expiration {
      days = 90
    }
  }
}