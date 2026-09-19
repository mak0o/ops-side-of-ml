# --- apply 用ロール（main へのプッシュからのみ引ける） ---

data "aws_iam_policy_document" "gha_apply_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    # main ブランチ限定。PR では引けない。
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${local.github_repo}:ref:refs/heads/main"]
    }
  }
}

resource "aws_iam_role" "gha_apply" {
  name               = "ops-side-of-ml-gha-apply"
  assume_role_policy = data.aws_iam_policy_document.gha_apply_assume.json
}

data "aws_iam_policy_document" "gha_apply" {
  # state へのフルアクセス
  statement {
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [local.state_bucket_arn]
  }

  statement {
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
    ]
    resources = ["${local.state_bucket_arn}/*"]
  }

  # 管理対象リソースの操作
  statement {
    effect = "Allow"
    actions = [
      "s3:*",
      "ecr:*",
      "sagemaker:*",
      "logs:*",
      "cloudwatch:*",
      "events:*",
      "lambda:*",
    ]
    resources = ["*"]
  }

  # IAM は SageMaker 実行ロールの管理に必要。
  # ただしプレフィクスを固定して、無関係なロールには触れないようにする。
  statement {
    effect = "Allow"
    actions = [
      "iam:CreateRole",
      "iam:DeleteRole",
      "iam:GetRole",
      "iam:PassRole",
      "iam:TagRole",
      "iam:AttachRolePolicy",
      "iam:DetachRolePolicy",
      "iam:PutRolePolicy",
      "iam:DeleteRolePolicy",
      "iam:GetRolePolicy",
      "iam:ListRolePolicies",
      "iam:ListAttachedRolePolicies",
      "iam:ListInstanceProfilesForRole",
    ]
    resources = ["arn:aws:iam::*:role/ops-side-of-ml-*"]
  }

  # 読み取りのみ。plan での差分検出に使う。
  statement {
    effect = "Allow"
    actions = [
      "iam:ListRoles",
      "iam:ListPolicies",
      "iam:GetPolicy",
      "iam:GetPolicyVersion",
      "iam:GetOpenIDConnectProvider",
      "sts:GetCallerIdentity",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "gha_apply" {
  name   = "terraform-apply"
  role   = aws_iam_role.gha_apply.id
  policy = data.aws_iam_policy_document.gha_apply.json
}

output "gha_plan_role_arn" {
  value = aws_iam_role.gha_plan.arn
}

output "gha_apply_role_arn" {
  value = aws_iam_role.gha_apply.arn
}