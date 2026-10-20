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

variable "model_artifact_s3_uri" {
  type        = string
  description = "S3 URI of the trained model.tar.gz (output of SageMaker training pipeline)"
}

variable "instance_type" {
  type    = string
  default = "ml.t2.medium"
}

variable "initial_instance_count" {
  type    = number
  default = 1
}
