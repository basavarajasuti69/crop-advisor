resource "aws_ecr_repository" "api" {
  name = "${local.name_prefix}-api"
  image_scanning_configuration {
    scan_on_push = true
  }
  tags = local.tags
}

resource "aws_cloudwatch_log_group" "api" {
  name = "/aws/ecs/${local.name_prefix}-api"
  retention_in_days = 30
  tags = local.tags
}

output "api_repository_url" {
  value = aws_ecr_repository.api.repository_url
}
