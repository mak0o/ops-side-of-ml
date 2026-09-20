locals {
  ecr_repos = ["train", "serve"]
}

resource "aws_ecr_repository" "this" {
  for_each = toset(local.ecr_repos)

  name = "${var.project}/${each.key}"

  image_scanning_configuration {
    scan_on_push = true
  }

  image_tag_mutability = "MUTABLE"
}

resource "aws_ecr_lifecycle_policy" "this" {
  for_each = aws_ecr_repository.this

  repository = each.value.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "タグなしイメージは 1 日で削除"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 1
        }
        action = { type = "expire" }
      },
      {
        rulePriority = 2
        description  = "タグ付きは直近 10 世代のみ保持"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = { type = "expire" }
      },
    ]
  })
}

output "ecr_urls" {
  value = { for k, v in aws_ecr_repository.this : k => v.repository_url }
}

# 単一リソースから for_each へ移行した。
# これが無いと train リポジトリが destroy / create になり、push 済みの
# イメージが失われる。移行が完了したら削除してよい。
moved {
  from = aws_ecr_repository.train
  to   = aws_ecr_repository.this["train"]
}

moved {
  from = aws_ecr_lifecycle_policy.train
  to   = aws_ecr_lifecycle_policy.this["train"]
}