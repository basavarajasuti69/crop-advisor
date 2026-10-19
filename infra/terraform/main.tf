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

module "network" {
  source               = "./network"
  region               = var.region
  project              = var.project
  environment          = var.environment
  vpc_cidr             = var.vpc_cidr
  public_subnet_cidrs  = var.public_subnet_cidrs
  private_subnet_cidrs = var.private_subnet_cidrs
}

module "ecs" {
  source      = "./ecs"
  region      = var.region
  project     = var.project
  environment = var.environment

  api_image          = var.api_image
  desired_count      = var.desired_count
  container_port     = var.container_port
  sagemaker_endpoint = module.sagemaker.endpoint_name
  guardrail_id       = module.sagemaker.guardrail_id

  vpc_id             = module.network.vpc_id
  public_subnet_ids  = module.network.public_subnet_ids
  private_subnet_ids = module.network.private_subnet_ids
}

output "api_url" {
  value = module.ecs.api_url
}

output "ecr_repository_url" {
  value = module.ecs.api_repository_url
}
