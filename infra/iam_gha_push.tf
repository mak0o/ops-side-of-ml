# --- ECR push 用ロール（main への push のみ） ---

data "aws_iam_policy_document" "gha_push_assume" {
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

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${local.github_repo}:ref:refs/heads/main"]
    }
  }
}

resource "aws_iam_role" "gha_push" {
  name               = "ops-side-of-ml-gha-push"
  assume_role_policy = data.aws_iam_policy_document.gha_push_assume.json
}

data "aws_iam_policy_document" "gha_push" {
  # ECR ログインはリソース指定ができない
  statement {
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  # push 先は train リポジトリのみ
  statement {
    effect = "Allow"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:CompleteLayerUpload",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:UploadLayerPart",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
    ]
    resources = [for r in aws_ecr_repository.this : r.arn]
  }
}

resource "aws_iam_role_policy" "gha_push" {
  name   = "ecr-push"
  role   = aws_iam_role.gha_push.id
  policy = data.aws_iam_policy_document.gha_push.json
}

output "gha_push_role_arn" {
  value = aws_iam_role.gha_push.arn
}