locals {
  endpoint_name = "${var.project}-endpoint"
  serve_image   = "${aws_ecr_repository.this["serve"].repository_url}:${var.serve_image_tag}"
}

# モデル定義。イメージと成果物の組み合わせを固定する。
resource "aws_sagemaker_model" "this" {
  # name を省略すると Terraform が一意な名前を生成する。
  # SageMaker のモデルは作成後に変更できないため、内容が変わったら
  # 新しい名前で作り直す必要がある。
  execution_role_arn = aws_iam_role.sagemaker_execution.arn

  primary_container {
    image          = local.serve_image
    model_data_url = var.model_artifact_uri

    environment = {
      INFERENCE_LOG_BUCKET     = aws_s3_bucket.artifacts.id
      INFERENCE_LOG_PREFIX     = "inference-logs"
      INFERENCE_LOG_FLUSH_SIZE = "1"
      MODEL_VERSION            = var.serve_image_tag
    }
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_sagemaker_endpoint_configuration" "this" {
  name_prefix = "${var.project}-"

  production_variants {
    variant_name           = "default"
    model_name             = aws_sagemaker_model.this.name
    instance_type          = "ml.t2.medium"
    initial_instance_count = 1
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_sagemaker_endpoint" "this" {
  name                 = local.endpoint_name
  endpoint_config_name = aws_sagemaker_endpoint_configuration.this.name
}

output "endpoint_name" {
  value = aws_sagemaker_endpoint.this.name
}