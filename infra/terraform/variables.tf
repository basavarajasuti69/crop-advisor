variable "region" {
  type        = string
  default     = "ap-south-1"
  description = "AWS region"
}

variable "project" {
  type    = string
  default = "crop-advisor"
}

variable "environment" {
  type        = string
  default     = "dev"
  description = "dev | prod"
}

variable "vpc_cidr" {
  type    = string
  default = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  type    = list(string)
  default = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "private_subnet_cidrs" {
  type    = list(string)
  default = ["10.0.10.0/24", "10.0.20.0/24"]
}

variable "api_image" {
  type        = string
  description = "ECR image URI for the API"
}

variable "desired_count" {
  type    = number
  default = 2
}

variable "container_port" {
  type    = number
  default = 8000
}
