# GitHub Actions 用の OIDC プロバイダ。アカウントに1つだけ存在すればよい。
resource "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"

  client_id_list = ["sts.amazonaws.com"]

  # AWS 側でルート CA を検証するため、thumbprint は実質使われない。
  # 属性自体は必須なので GitHub の現行値を入れておく。
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

locals {
  github_repo = "mak0o/ops-side-of-ml"

  # state バケット。bootstrap で作ったものを名前で参照する。
  state_bucket_arn = "arn:aws:s3:::ops-side-of-ml-tfstate"
}