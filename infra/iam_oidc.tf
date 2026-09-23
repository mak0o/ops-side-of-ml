# GitHub Actions 用の OIDC プロバイダ。アカウントに1つだけ存在すればよい。
# 使わない期間は enable_github_oidc = false で削除し、GitHub から AWS への経路を塞ぐ。
# ロール自体は残るので、戻すのは apply 1 回で済む。
resource "aws_iam_openid_connect_provider" "github" {
  count = var.enable_github_oidc ? 1 : 0

  url = "https://token.actions.githubusercontent.com"

  client_id_list = ["sts.amazonaws.com"]

  # AWS 側でルート CA を検証するため、thumbprint は実質使われない。
  # 属性自体は必須なので GitHub の現行値を入れておく。
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

locals {
  github_repo = "mak0o@36266249/ops-side-of-ml@1366131270"

  # 信頼ポリシーではリソースを参照せず、ARN を組み立てる。
  # 参照すると、プロバイダを消したときにロールの信頼ポリシーも変更扱いになり、
  # 戻すたびに 3 つのロールが作り直しになる。
  github_oidc_provider_arn = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/token.actions.githubusercontent.com"

  # state バケット。bootstrap で作ったものを名前で参照する。
  state_bucket_arn = "arn:aws:s3:::ops-side-of-ml-tfstate"
}