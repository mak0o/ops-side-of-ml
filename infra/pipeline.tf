# 再学習パイプライン。
#
#   ResolveModel → CreateModel → Transform → DeleteModel → DriftCheck
#     → (ドリフトあり) Train → RegisterAndPromote
#
# 推論データ（input_prefix）と再学習データ（training_input_prefix）は実行時の入力で受け取る。
# データを S3 に置く役はこのワークフローの外にある。
#
# ジョブ名には実行名を使う。既定の実行名（UUID）を前提にしているので、
# 手動実行で名前を付けるときは英数字とハイフンだけにすること。

locals {
  pipeline_name       = "${var.project}-pipeline"
  bucket              = aws_s3_bucket.artifacts.id
  account_id          = data.aws_caller_identity.current.account_id
  model_package_group = aws_sagemaker_model_package_group.this.model_package_group_name

  # drift_check / register_and_promote は serve イメージで動かす（boto3 が入っているため）
  pipeline_serve_image = "${aws_ecr_repository.this["serve"].repository_url}:${var.pipeline_image_tag}"
  pipeline_train_image = "${aws_ecr_repository.this["train"].repository_url}:${var.pipeline_image_tag}"

  # scripts/train_sagemaker.py と同じ定義。average_precision の誤検出を防ぐため行頭で固定する。
  metric_definitions = [
    for name in ["precision", "recall", "f1", "f2", "average_precision"] :
    { Name = name, Regex = "^${name}=([0-9\\.]+);" }
  ]

  processing_resources = {
    ClusterConfig = {
      InstanceCount  = 1
      InstanceType   = "ml.m5.large"
      VolumeSizeInGB = 10
    }
  }

  # 実行開始時刻 "2026-09-21T05:12:34.567Z" を ["2026", "09", "21"] に分解する
  date_parts = "States.StringSplit(States.ArrayGetItem(States.StringSplit($$.Execution.StartTime, 'T'), 0), '-')"
}

# ---------------------------------------------------------------------------
# SageMaker 実行ロールへの追加権限
# Processing Job の中で drift_check と register_and_promote が Model Registry を操作する。
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "sagemaker_registry" {
  statement {
    sid = "ModelRegistry"
    actions = [
      "sagemaker:DescribeModelPackage",
      "sagemaker:CreateModelPackage",
      "sagemaker:UpdateModelPackage",
    ]
    resources = [
      aws_sagemaker_model_package_group.this.arn,
      "arn:aws:sagemaker:${var.region}:${local.account_id}:model-package/${local.model_package_group}/*",
    ]
  }

  # List 系はリソース指定ができない
  statement {
    sid       = "ListModelPackages"
    actions   = ["sagemaker:ListModelPackages"]
    resources = ["*"]
  }

  statement {
    sid       = "DescribeTrainingJob"
    actions   = ["sagemaker:DescribeTrainingJob"]
    resources = ["arn:aws:sagemaker:${var.region}:${local.account_id}:training-job/*"]
  }
}

resource "aws_iam_role_policy" "sagemaker_registry" {
  name   = "model-registry"
  role   = aws_iam_role.sagemaker_execution.id
  policy = data.aws_iam_policy_document.sagemaker_registry.json
}

# ---------------------------------------------------------------------------
# Step Functions のロール
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "sfn_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["states.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }
  }
}

resource "aws_iam_role" "sfn" {
  name               = "${var.project}-sfn"
  assume_role_policy = data.aws_iam_policy_document.sfn_assume.json
}

data "aws_iam_policy_document" "sfn" {
  # ジョブ名の接頭辞でリソースを絞る。SageMaker の ARN ではジョブ名が小文字になる。
  statement {
    sid = "SageMakerJobs"
    actions = [
      "sagemaker:CreateTransformJob",
      "sagemaker:DescribeTransformJob",
      "sagemaker:StopTransformJob",
      "sagemaker:CreateTrainingJob",
      "sagemaker:DescribeTrainingJob",
      "sagemaker:StopTrainingJob",
      "sagemaker:CreateProcessingJob",
      "sagemaker:DescribeProcessingJob",
      "sagemaker:StopProcessingJob",
      "sagemaker:CreateModel",
      "sagemaker:DeleteModel",
      # .sync 統合はジョブに管理用のタグを自動で付けるので必要
      "sagemaker:AddTags",
    ]
    resources = [
      "arn:aws:sagemaker:${var.region}:${local.account_id}:transform-job/bt-*",
      "arn:aws:sagemaker:${var.region}:${local.account_id}:training-job/tr-*",
      "arn:aws:sagemaker:${var.region}:${local.account_id}:processing-job/dc-*",
      "arn:aws:sagemaker:${var.region}:${local.account_id}:processing-job/rp-*",
      "arn:aws:sagemaker:${var.region}:${local.account_id}:model/md-*",
    ]
  }

  statement {
    sid     = "ResolveModel"
    actions = ["sagemaker:DescribeModelPackage"]
    resources = [
      "arn:aws:sagemaker:${var.region}:${local.account_id}:model-package/${local.model_package_group}/*",
    ]
  }

  statement {
    sid       = "ListModelPackages"
    actions   = ["sagemaker:ListModelPackages"]
    resources = ["*"]
  }

  # ジョブに実行ロールを渡す。渡し先を SageMaker に限定する。
  statement {
    sid       = "PassExecutionRole"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.sagemaker_execution.arn]

    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["sagemaker.amazonaws.com"]
    }
  }

  # .sync（完了待ち）の統合は、Step Functions が管理する EventBridge ルールでジョブの完了を受け取る
  statement {
    sid = "SyncIntegrationRules"
    actions = [
      "events:PutTargets",
      "events:PutRule",
      "events:DescribeRule",
    ]
    resources = [
      "arn:aws:events:${var.region}:${local.account_id}:rule/StepFunctionsGetEventsForSageMakerTransformJobsRule",
      "arn:aws:events:${var.region}:${local.account_id}:rule/StepFunctionsGetEventsForSageMakerTrainingJobsRule",
      "arn:aws:events:${var.region}:${local.account_id}:rule/StepFunctionsGetEventsForSageMakerProcessingJobsRule",
    ]
  }

  # Processing Job が書いた判定結果を読む
  statement {
    sid       = "ReadResults"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.artifacts.arn}/pipeline-results/*"]
  }
}

resource "aws_iam_role_policy" "sfn" {
  name   = "pipeline"
  role   = aws_iam_role.sfn.id
  policy = data.aws_iam_policy_document.sfn.json
}

# ---------------------------------------------------------------------------
# ステートマシン
# ---------------------------------------------------------------------------

resource "aws_sfn_state_machine" "pipeline" {
  name     = local.pipeline_name
  role_arn = aws_iam_role.sfn.arn

  definition = jsonencode({
    Comment = "バッチ推論 → ドリフト検知 → 再学習 → 昇格判定"
    StartAt = "Prepare"

    States = {

      # --- 実行ごとの名前とパスを組み立てる ---

      Prepare = {
        Type = "Pass"
        Parameters = {
          "name.$" = "$$.Execution.Name"
          "date.$" = local.date_parts
        }
        ResultPath = "$.run"
        Next       = "BuildPaths"
      }

      BuildPaths = {
        Type = "Pass"
        Parameters = {
          # 既存の推論ログと同じ日付パーティションに出す
          "logs_prefix.$" = "States.Format('inference-logs/year={}/month={}/day={}/bt-{}/', States.ArrayGetItem($.run.date, 0), States.ArrayGetItem($.run.date, 1), States.ArrayGetItem($.run.date, 2), $.run.name)"
          "drift_key.$"   = "States.Format('pipeline-results/{}/drift.json', $.run.name)"
          "promote_key.$" = "States.Format('pipeline-results/{}/promote.json', $.run.name)"
        }
        ResultPath = "$.paths"
        Next       = "ResolveModel"
      }

      # --- 現行モデル（最新の Approved）を解決する。src/registry.py の latest_approved と同じ定義 ---

      ResolveModel = {
        Type     = "Task"
        Resource = "arn:aws:states:::aws-sdk:sagemaker:listModelPackages"
        Parameters = {
          ModelPackageGroupName = local.model_package_group
          ModelApprovalStatus   = "Approved"
          SortBy                = "CreationTime"
          SortOrder             = "Descending"
          MaxResults            = 1
        }
        ResultSelector = {
          "packages.$" = "$.ModelPackageSummaryList"
        }
        ResultPath = "$.resolved"
        Next       = "HasApprovedModel"
      }

      HasApprovedModel = {
        Type = "Choice"
        Choices = [
          {
            Variable  = "$.resolved.packages[0]"
            IsPresent = true
            Next      = "DescribeModel"
          }
        ]
        Default = "NoApprovedModel"
      }

      DescribeModel = {
        Type     = "Task"
        Resource = "arn:aws:states:::aws-sdk:sagemaker:describeModelPackage"
        Parameters = {
          "ModelPackageName.$" = "$.resolved.packages[0].ModelPackageArn"
        }
        ResultSelector = {
          "arn.$"            = "$.ModelPackageArn"
          "version.$"        = "$.ModelPackageVersion"
          "image.$"          = "$.InferenceSpecification.Containers[0].Image"
          "model_data_url.$" = "$.InferenceSpecification.Containers[0].ModelDataUrl"
        }
        ResultPath = "$.model"
        Next       = "CreateModel"
      }

      # --- バッチ推論。Model は Transform Job と同じライフサイクルで作って消す ---

      CreateModel = {
        Type     = "Task"
        Resource = "arn:aws:states:::sagemaker:createModel"
        Parameters = {
          "ModelName.$"    = "States.Format('md-{}', $.run.name)"
          ExecutionRoleArn = aws_iam_role.sagemaker_execution.arn
          PrimaryContainer = {
            "Image.$"        = "$.model.image"
            "ModelDataUrl.$" = "$.model.model_data_url"
            Environment = {
              # 出力は SageMaker が S3 に書くので自前のログは止める
              INFERENCE_LOG_ENABLED = "0"
              "MODEL_VERSION.$"     = "States.Format('${local.model_package_group}/{}', $.model.version)"
            }
          }
        }
        ResultPath = null
        Next       = "Transform"
      }

      Transform = {
        Type     = "Task"
        Resource = "arn:aws:states:::sagemaker:createTransformJob.sync"
        Parameters = {
          "TransformJobName.$" = "States.Format('bt-{}', $.run.name)"
          "ModelName.$"        = "States.Format('md-{}', $.run.name)"
          TransformInput = {
            DataSource = {
              S3DataSource = {
                S3DataType = "S3Prefix"
                "S3Uri.$"  = "$.input_prefix"
              }
            }
            # FastAPI は application/json 以外を受け付けない。1 行ずつ切り出して JSON として送る。
            ContentType = "application/json"
            SplitType   = "Line"
          }
          TransformOutput = {
            "S3OutputPath.$" = "States.Format('s3://${local.bucket}/{}', $.paths.logs_prefix)"
            Accept           = "application/json"
            AssembleWith     = "Line"
          }
          TransformResources = {
            InstanceType  = "ml.m5.large"
            InstanceCount = 1
          }
          BatchStrategy = "SingleRecord"
          # 出力に特徴量を残す。drift_check が分布を比べるのに要る。
          DataProcessing = {
            InputFilter  = "$"
            JoinSource   = "Input"
            OutputFilter = "$"
          }
        }
        ResultPath = null
        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            ResultPath  = "$.error"
            Next        = "CleanupAfterTransformFailure"
          }
        ]
        Next = "DeleteModel"
      }

      DeleteModel = {
        Type     = "Task"
        Resource = "arn:aws:states:::aws-sdk:sagemaker:deleteModel"
        Parameters = {
          "ModelName.$" = "States.Format('md-{}', $.run.name)"
        }
        ResultPath = null
        Next       = "DriftCheck"
      }

      # Transform が失敗しても Model は必ず消す
      CleanupAfterTransformFailure = {
        Type     = "Task"
        Resource = "arn:aws:states:::aws-sdk:sagemaker:deleteModel"
        Parameters = {
          "ModelName.$" = "States.Format('md-{}', $.run.name)"
        }
        ResultPath = null
        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            ResultPath  = "$.cleanup_error"
            Next        = "TransformFailed"
          }
        ]
        Next = "TransformFailed"
      }

      # --- ドリフト検知。判定不能とクラッシュだけがジョブ失敗になる ---

      DriftCheck = {
        Type     = "Task"
        Resource = "arn:aws:states:::sagemaker:createProcessingJob.sync"
        Parameters = {
          "ProcessingJobName.$" = "States.Format('dc-{}', $.run.name)"
          RoleArn               = aws_iam_role.sagemaker_execution.arn
          AppSpecification = {
            ImageUri               = local.pipeline_serve_image
            ContainerEntrypoint    = ["python", "-m", "src.drift_check"]
            "ContainerArguments.$" = "States.Array('--model-package-group', '${local.model_package_group}', '--prefix', $.paths.logs_prefix, '--result-s3-uri', States.Format('s3://${local.bucket}/{}', $.paths.drift_key))"
          }
          ProcessingResources = local.processing_resources
          Environment = {
            INFERENCE_LOG_BUCKET = local.bucket
            AWS_DEFAULT_REGION   = var.region
          }
          StoppingCondition = {
            MaxRuntimeInSeconds = 1800
          }
        }
        ResultPath = null
        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            ResultPath  = "$.error"
            Next        = "DriftCheckFailed"
          }
        ]
        Next = "ReadDriftResult"
      }

      ReadDriftResult = {
        Type     = "Task"
        Resource = "arn:aws:states:::aws-sdk:s3:getObject"
        Parameters = {
          Bucket  = local.bucket
          "Key.$" = "$.paths.drift_key"
        }
        ResultSelector = {
          "result.$" = "States.StringToJson($.Body)"
        }
        ResultPath = "$.drift"
        Next       = "IsDrift"
      }

      IsDrift = {
        Type = "Choice"
        Choices = [
          {
            Variable      = "$.drift.result.drift"
            BooleanEquals = true
            Next          = "Train"
          }
        ]
        Default = "NoDrift"
      }

      NoDrift = {
        Type = "Succeed"
      }

      # --- 再学習 ---

      Train = {
        Type     = "Task"
        Resource = "arn:aws:states:::sagemaker:createTrainingJob.sync"
        Parameters = {
          "TrainingJobName.$" = "States.Format('tr-{}', $.run.name)"
          AlgorithmSpecification = {
            TrainingImage     = local.pipeline_train_image
            TrainingInputMode = "File"
            MetricDefinitions = local.metric_definitions
          }
          RoleArn = aws_iam_role.sagemaker_execution.arn
          InputDataConfig = [
            {
              ChannelName = "train"
              DataSource = {
                S3DataSource = {
                  S3DataType             = "S3Prefix"
                  "S3Uri.$"              = "$.training_input_prefix"
                  S3DataDistributionType = "FullyReplicated"
                }
              }
            }
          ]
          OutputDataConfig = {
            S3OutputPath = "s3://${local.bucket}/training-output/"
          }
          ResourceConfig = {
            InstanceType   = "ml.m5.large"
            InstanceCount  = 1
            VolumeSizeInGB = 10
          }
          HyperParameters = {
            n_estimators = "200"
            max_depth    = "8"
            since        = ""
          }
          StoppingCondition = {
            MaxRuntimeInSeconds = 1800
          }
        }
        ResultPath = null
        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            ResultPath  = "$.error"
            Next        = "TrainFailed"
          }
        ]
        Next = "RegisterAndPromote"
      }

      # --- 登録と昇格判定。Approved / Rejected のどちらもジョブは成功で終わる ---

      RegisterAndPromote = {
        Type     = "Task"
        Resource = "arn:aws:states:::sagemaker:createProcessingJob.sync"
        Parameters = {
          "ProcessingJobName.$" = "States.Format('rp-{}', $.run.name)"
          RoleArn               = aws_iam_role.sagemaker_execution.arn
          AppSpecification = {
            ImageUri               = local.pipeline_serve_image
            ContainerEntrypoint    = ["python", "-m", "src.register_and_promote"]
            "ContainerArguments.$" = "States.Array('--training-job', States.Format('tr-{}', $.run.name), '--serve-image', '${local.pipeline_serve_image}', '--eval-s3-uri', $.eval_prefix, '--result-s3-uri', States.Format('s3://${local.bucket}/{}', $.paths.promote_key))"
          }
          ProcessingResources = local.processing_resources
          Environment = {
            AWS_DEFAULT_REGION = var.region
          }
          StoppingCondition = {
            MaxRuntimeInSeconds = 1800
          }
        }
        ResultPath = null
        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            ResultPath  = "$.error"
            Next        = "PromoteFailed"
          }
        ]
        Next = "ReadPromoteResult"
      }

      ReadPromoteResult = {
        Type     = "Task"
        Resource = "arn:aws:states:::aws-sdk:s3:getObject"
        Parameters = {
          Bucket  = local.bucket
          "Key.$" = "$.paths.promote_key"
        }
        ResultSelector = {
          "result.$" = "States.StringToJson($.Body)"
        }
        ResultPath = "$.promote"
        Next       = "IsApproved"
      }

      IsApproved = {
        Type = "Choice"
        Choices = [
          {
            Variable      = "$.promote.result.approved"
            BooleanEquals = true
            Next          = "Promoted"
          }
        ]
        Default = "Rejected"
      }

      # どちらも正常な判定結果なので Succeed。状態名で結果を区別する。
      Promoted = {
        Type = "Succeed"
      }

      Rejected = {
        Type = "Succeed"
      }

      # --- 失敗 ---

      NoApprovedModel = {
        Type  = "Fail"
        Error = "NoApprovedModel"
        Cause = "Model Package Group に Approved のモデルがありません"
      }

      # Catch で $.error に保存した実際のエラーをそのまま出す。
      # 固定の文言にすると、原因を知るのに実行履歴を掘る必要がある。
      TransformFailed = {
        Type      = "Fail"
        ErrorPath = "$.error.Error"
        CausePath = "$.error.Cause"
      }

      DriftCheckFailed = {
        Type      = "Fail"
        ErrorPath = "$.error.Error"
        CausePath = "$.error.Cause"
      }

      TrainFailed = {
        Type      = "Fail"
        ErrorPath = "$.error.Error"
        CausePath = "$.error.Cause"
      }

      PromoteFailed = {
        Type      = "Fail"
        ErrorPath = "$.error.Error"
        CausePath = "$.error.Cause"
      }
    }
  })
}

# ---------------------------------------------------------------------------
# 日次スケジュール。動作確認が済むまで無効化しておく。
# 有効化すると Transform と DriftCheck で 1 日数円の課金が発生する。
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "scheduler_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["scheduler.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }
  }
}

resource "aws_iam_role" "scheduler" {
  name               = "${var.project}-scheduler"
  assume_role_policy = data.aws_iam_policy_document.scheduler_assume.json
}

data "aws_iam_policy_document" "scheduler" {
  statement {
    actions   = ["states:StartExecution"]
    # ステートマシンの属性を参照すると、定義を変えるたびに plan でこのポリシーも
    # 変更扱いになる。名前は固定なので ARN を組み立てて依存を切る。
    resources = ["arn:aws:states:${var.region}:${local.account_id}:stateMachine:${local.pipeline_name}"]
  }
}

resource "aws_iam_role_policy" "scheduler" {
  name   = "start-pipeline"
  role   = aws_iam_role.scheduler.id
  policy = data.aws_iam_policy_document.scheduler.json
}

resource "aws_scheduler_schedule" "pipeline" {
  name  = "${var.project}-daily"
  state = "DISABLED"

  schedule_expression          = "cron(0 10 * * ? *)"
  schedule_expression_timezone = "Asia/Tokyo"

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = aws_sfn_state_machine.pipeline.arn
    role_arn = aws_iam_role.scheduler.arn

    # データの配置はワークフローの外。上流が latest/ を毎日置き換える前提。
    input = jsonencode({
      input_prefix          = "s3://${local.bucket}/batch-input/latest/"
      training_input_prefix = "s3://${local.bucket}/training-input/latest/"
      eval_prefix           = "s3://${local.bucket}/eval-input/latest/"
    })
  }
}

output "pipeline_state_machine_arn" {
  value = aws_sfn_state_machine.pipeline.arn
}
