# SageMaker Model Package Group (Model Registry)
resource "aws_sagemaker_model_package_group" "classifier" {
  model_package_group_name        = "${local.name_prefix}-classifier"
  model_package_group_description = "Crop disease classifier model registry"
  tags                            = local.tags
}

# IAM role for SageMaker Pipeline
resource "aws_iam_role" "sagemaker_pipeline" {
  name = "${local.name_prefix}-sagemaker-pipeline"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      },
    ]
  })
  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "pipeline_sagemaker" {
  role       = aws_iam_role.sagemaker_pipeline.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_role_policy_attachment" "pipeline_s3" {
  role       = aws_iam_role.sagemaker_pipeline.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_role_policy_attachment" "pipeline_cloudwatch" {
  role       = aws_iam_role.sagemaker_pipeline.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchFullAccess"
}

output "model_package_group_name" {
  value = aws_sagemaker_model_package_group.classifier.model_package_group_name
}

output "pipeline_role_arn" {
  value = aws_iam_role.sagemaker_pipeline.arn
}
