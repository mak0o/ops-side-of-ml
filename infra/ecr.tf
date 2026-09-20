resource "aws_ecr_repository" "train" {
  name = "${var.project}/train"

  # push のたびに脆弱性スキャンを走らせる
  image_scanning_configuration {
    scan_on_push = true
  }

  # タグの上書きを許す。CI から latest を更新するため。
  image_tag_mutability = "MUTABLE"
}

# イメージは放置すると溜まり続けて課金対象になる
resource "aws_ecr_lifecycle_policy" "train" {
  repository = aws_ecr_repository.train.name

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

output "ecr_train_url" {
  value = aws_ecr_repository.train.repository_url
}