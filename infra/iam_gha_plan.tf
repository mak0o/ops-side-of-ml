# --- plan 用ロール（PR から引ける） ---

data "aws_iam_policy_document" "gha_plan_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [local.github_oidc_provider_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    # PR は任意のブランチから作られるため ref を限定できない。
    # 権限が ReadOnly + state 書き込みに限られることが安全性の担保。
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${local.github_repo}:*"]
    }
  }
}

resource "aws_iam_role" "gha_plan" {
  name               = "ops-side-of-ml-gha-plan"
  assume_role_policy = data.aws_iam_policy_document.gha_plan_assume.json
}

# 読み取りは AWS 管理ポリシーに任せる。自前で列挙すると plan のたびに
# 権限不足で落ちて、その都度追記することになる。
resource "aws_iam_role_policy_attachment" "gha_plan_readonly" {
  role       = aws_iam_role.gha_plan.name
  policy_arn = "arn:aws:iam::aws:policy/ReadOnlyAccess"
}

# plan は refresh で state を更新するため、state への書き込みが要る。
data "aws_iam_policy_document" "gha_plan_state" {
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
}

resource "aws_iam_role_policy" "gha_plan_state" {
  name   = "state-access"
  role   = aws_iam_role.gha_plan.id
  policy = data.aws_iam_policy_document.gha_plan_state.json
}