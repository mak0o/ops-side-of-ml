variable "region" {
  type    = string
  default = "ap-northeast-1"
}

variable "project" {
  type    = string
  default = "ops-side-of-ml"
}

variable "model_artifact_uri" {
  type        = string
  description = "デプロイする model.tar.gz の S3 URI。学習ジョブの成果物を明示的に指定する。"
}

variable "serve_image_tag" {
  type        = string
  description = "推論イメージのタグ。latest だと Terraform が変更を検知できないためコミットハッシュを推奨。"
  default     = "latest"
}

variable "pipeline_image_tag" {
  type        = string
  description = "パイプラインが使う train / serve イメージのタグ。image ジョブは同じコミットで両方をビルドするので1つで足りる。"
}

variable "enable_github_oidc" {
  type        = bool
  description = "GitHub Actions から AWS を操作できるようにするか。false にすると OIDC プロバイダを削除し、3 つのロールを誰も引き受けられなくする（ロール自体は残る）"
  default     = true
}
