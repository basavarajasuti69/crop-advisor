data "aws_ecr_repository" "api" {
  name = "${var.project}-${var.environment}-api"
}
