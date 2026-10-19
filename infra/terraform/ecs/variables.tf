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

variable "api_image" {
  type        = string
  description = "ECR image URI for the API"
}

variable "sagemaker_endpoint" {
  type        = string
  default     = ""
  description = "SageMaker endpoint serving the classifier (empty falls back to the local model)"
}

variable "guardrail_id" {
  type        = string
  default     = ""
  description = "Bedrock guardrail id used to screen the advice text"
}

variable "desired_count" {
  type    = number
  default = 2
}

variable "container_port" {
  type    = number
  default = 8000
}

variable "vpc_id" {
  type        = string
  description = "VPC the service runs in"
}

variable "public_subnet_ids" {
  type        = list(string)
  description = "Subnets for the load balancer"
}

variable "private_subnet_ids" {
  type        = list(string)
  description = "Subnets for the tasks"
}

variable "log_retention_days" {
  type    = number
  default = 30
}
