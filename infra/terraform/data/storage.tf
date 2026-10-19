resource "aws_s3_bucket" "images" {
  bucket = "${local.name_prefix}-images"
  tags   = local.tags
}

resource "aws_s3_bucket_public_access_block" "images" {
  bucket              = aws_s3_bucket.images.id
  block_public_acls   = true
  block_public_policy = true
}

resource "aws_dynamodb_table" "cases" {
  name         = "${local.name_prefix}-cases"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "case_id"
  range_key    = "sk"

  attribute {
    name = "case_id"
    type = "S"
  }
  attribute {
    name = "sk"
    type = "S"
  }

  tags = local.tags
}

output "images_bucket" {
  value = aws_s3_bucket.images.id
}

output "cases_table" {
  value = aws_dynamodb_table.cases.name
}
