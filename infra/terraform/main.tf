terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

locals {
  name_prefix = "${var.project}-${var.environment}"
  tags = {
    project     = var.project
    environment = var.environment
    managed_by  = "terraform"
  }
}

module "sagemaker" {
  source                = "./sagemaker"
  region                = var.region
  project               = var.project
  environment           = var.environment
  model_artifact_s3_uri = "s3://${local.name_prefix}-models/model.tar.gz"
}
