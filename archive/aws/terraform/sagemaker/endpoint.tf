# SageMaker model from trained artifact
resource "aws_sagemaker_model" "classifier" {
  name = "${local.name_prefix}-classifier"

  primary_container {
    image          = data.aws_ecr_repository.api.repository_url
    model_data_url = var.model_artifact_s3_uri
    environment = {
      CA_REGION = var.region
    }
  }

  execution_role_arn = aws_iam_role.sagemaker.arn
  tags               = local.tags
}

# Endpoint configuration
resource "aws_sagemaker_endpoint_configuration" "classifier" {
  name = "${local.name_prefix}-classifier-config"

  production_variants {
    variant_name           = "primary"
    model_name             = aws_sagemaker_model.classifier.name
    instance_type          = var.instance_type
    initial_instance_count = var.initial_instance_count
  }

  tags = local.tags
}

# Endpoint
resource "aws_sagemaker_endpoint" "classifier" {
  name                 = "${local.name_prefix}-classifier"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.classifier.name

  tags = local.tags
}

output "endpoint_name" {
  value = aws_sagemaker_endpoint.classifier.name
}

output "endpoint_arn" {
  value = aws_sagemaker_endpoint.classifier.arn
}
