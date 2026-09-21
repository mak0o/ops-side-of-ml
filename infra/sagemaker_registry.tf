# SageMaker Model Registry。ローカルの MLflow Registered Model に相当する。
# グループは継続的に存在するので Terraform で管理し、
# 中身の Model Package（バージョン）は学習のたびにスクリプトが登録する。
resource "aws_sagemaker_model_package_group" "this" {
  # MLflow の登録名と揃える
  model_package_group_name        = "cost-anomaly-detector"
  model_package_group_description = "AWS コスト異常検知モデル。最新の Approved を現行とする。"
}

output "model_package_group_name" {
  value = aws_sagemaker_model_package_group.this.model_package_group_name
}