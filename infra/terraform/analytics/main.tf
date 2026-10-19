# Outbreak Analytics — Athena + QuickSight infrastructure

variable "region" {
  type    = string
  default = "ap-south-1"
}

variable "project" {
  type    = string
  default = "crop-advisor"
}

variable "environment" {
  type    = string
  default = "dev"
}

variable "analytics_bucket" {
  type        = string
  description = "S3 bucket for Athena query results and case data"
}

variable "glue_database_name" {
  type    = string
  default = "crop_advisor_analytics"
}

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "region" { type = string }
variable "project" { type = string }
variable "environment" { type = string }
variable "analytics_bucket" { type = string }
variable "glue_database_name" { type = string }

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

# S3 bucket for analytics data + Athena query results
resource "aws_s3_bucket" "analytics" {
  bucket = var.analytics_bucket
  tags   = local.tags
}

resource "aws_s3_bucket_public_access_block" "analytics" {
  bucket              = aws_s3_bucket.analytics.id
  block_public_acls   = true
  block_public_policy = true
}

# Glue database for Athena
resource "aws_glue_catalog_database" "analytics" {
  name = var.glue_database_name
  tags = local.tags
}

# Glue crawler for cases table (scans S3 case data)
resource "aws_glue_crawler" "cases" {
  name          = "${local.name_prefix}-cases-crawler"
  role_arn      = aws_iam_role.glue.arn
  database_name = aws_glue_catalog_database.analytics.name
  targets {
    s3_targets {
      path = "s3://${var.analytics_bucket}/cases/"
    }
  }
  schedule {
    schedule_expression = "cron(0 */6 * * ? *)" # every 6 hours
  }
  tags = local.tags
}

# IAM role for Glue crawler
resource "aws_iam_role" "glue" {
  name = "${local.name_prefix}-glue-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = { Service = "glue.amazonaws.com" }
        Action    = "sts:AssumeRole"
      },
    ]
  })
  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "glue" {
  role       = aws_iam_role.glue.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole"
}

resource "aws_iam_role_policy_attachment" "glue_s3" {
  role       = aws_iam_role.glue.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

# Athena workgroup
resource "aws_athena_workgroup" "analytics" {
  name        = "${local.name_prefix}-analytics"
  description = "Outbreak analytics workgroup for crop disease advisor"

  configuration {
    result_configuration {
      output_location = "s3://${var.analytics_bucket}/athena-results/"
      encryption_configuration {
        encryption_option = "SSE_S3"
      }
    }
    enforce_workgroup_configuration = true
    publish_cloudwatch_metrics      = true
  }
  tags = local.tags
}

output "analytics_bucket" {
  value = aws_s3_bucket.analytics.id
}

output "glue_database" {
  value = aws_glue_catalog_database.analytics.name
}

output "athena_workgroup" {
  value = aws_athena_workgroup.analytics.name
}

output "glue_crawler" {
  value = aws_glue_crawler.cases.name
}
