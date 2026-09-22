# 通知。パイプラインの失敗、現行モデルの劣化、モデルの入れ替えを SNS に送る。
#
# 購読（メールアドレス）は Terraform で管理しない。公開リポジトリにアドレスを書けないのと、
# メールの購読は受信者が確認リンクを押すまで保留になり、Terraform では完了できないため。
#
#   aws sns subscribe --topic-arn <arn> --protocol email --notification-endpoint <address>
#
# 暗号化はしない。AWS 管理キー（alias/aws/sns）では EventBridge から送れず、
# カスタマー管理キーは月 1 ドルかかる。本文は実行名と判定の数値だけで機密ではない。

locals {
  # ステートマシンの属性を参照すると、定義を変えるたびに plan でルールも変更扱いになる。
  # 名前は固定なので ARN を組み立てて依存を切る（scheduler のポリシーと同じ理由）。
  pipeline_arn = "arn:aws:states:${var.region}:${local.account_id}:stateMachine:${local.pipeline_name}"
}

resource "aws_sns_topic" "alerts" {
  name = "${var.project}-alerts"
}

# EventBridge からの送信を許可する。送信元をこのルールに限定する。
# Step Functions からの送信は、同一アカウントなので SFN ロールの IAM ポリシーだけで足りる。
data "aws_iam_policy_document" "alerts" {
  statement {
    sid       = "AllowPipelineFailedRule"
    actions   = ["sns:Publish"]
    resources = [aws_sns_topic.alerts.arn]

    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }

    condition {
      test     = "ArnEquals"
      variable = "aws:SourceArn"
      values   = [aws_cloudwatch_event_rule.pipeline_failed.arn]
    }
  }
}

resource "aws_sns_topic_policy" "alerts" {
  arn    = aws_sns_topic.alerts.arn
  policy = data.aws_iam_policy_document.alerts.json
}

# パイプラインの失敗はステートマシンの外で拾う。
# 中に書くと、ステートマシン自体が壊れたときに通知も一緒に止まる。
resource "aws_cloudwatch_event_rule" "pipeline_failed" {
  name        = "${var.project}-pipeline-failed"
  description = "パイプラインが FAILED / TIMED_OUT / ABORTED で終わったら通知する"

  event_pattern = jsonencode({
    source        = ["aws.states"]
    "detail-type" = ["Step Functions Execution Status Change"]
    detail = {
      stateMachineArn = [local.pipeline_arn]
      status          = ["FAILED", "TIMED_OUT", "ABORTED"]
    }
  })
}

resource "aws_cloudwatch_event_target" "pipeline_failed" {
  rule = aws_cloudwatch_event_rule.pipeline_failed.name
  arn  = aws_sns_topic.alerts.arn

  # 原因は Fail State が実行結果に出すので、describe-execution のコマンドを本文に入れる
  input_transformer {
    input_paths = {
      status = "$.detail.status"
      name   = "$.detail.name"
      arn    = "$.detail.executionArn"
    }
    input_template = "\"パイプラインが <status> で終了しました。実行名: <name> / 原因の確認: aws stepfunctions describe-execution --execution-arn <arn> --query [error,cause]\""
  }
}

output "alerts_topic_arn" {
  value = aws_sns_topic.alerts.arn
}