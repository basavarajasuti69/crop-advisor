# IAM for the ECS service: the execution role pulls the image, the task role
# is what the container itself may call.

resource "aws_iam_role" "execution" {
  name = "${local.name_prefix}-api-execution"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "execution_ecr" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "task" {
  name = "${local.name_prefix}-api-task"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
  tags = local.tags
}

resource "aws_iam_role_policy" "task_logs" {
  name = "${local.name_prefix}-api-task-logs"
  role = aws_iam_role.task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["logs:CreateLogStream", "logs:PutLogEvents"]
      Resource = "arn:aws:logs:${var.region}:*:log-group:/aws/ecs/${local.name_prefix}-api*"
    }]
  })
}

resource "aws_iam_role_policy" "task_inference" {
  name = "${local.name_prefix}-api-task-inference"
  role = aws_iam_role.task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "sagemaker:InvokeEndpoint",
        "sagemaker:DescribeEndpoint",
        "bedrock:ApplyGuardrail",
        "bedrock:InvokeModel",
      ]
      Resource = "*"
    }]
  })
}

resource "aws_iam_role_policy" "task_s3" {
  name = "${local.name_prefix}-api-task-s3"
  role = aws_iam_role.task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:ListBucket"]
      Resource = ["arn:aws:s3:::${local.name_prefix}-*", "arn:aws:s3:::${local.name_prefix}-*/*"]
    }]
  })
}

output "task_role_arn" {
  value = aws_iam_role.task.arn
}

output "execution_role_arn" {
  value = aws_iam_role.execution.arn
}
